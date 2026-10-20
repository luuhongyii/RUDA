use super::{DataLoader, DataLoaderIterator, Progress, batcher::Batcher};
use ruda_core::rand::{CheckpointRng, SeedableRng, checkpoint_rng, restore_rng};
use ruda_dataset::{Dataset, transform::PartialDataset};
use ruda_tensor::{AutodiffContext, backend::Backend};
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::{sync::{Arc, Mutex, mpsc}, thread};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DataLoaderState {
    pub version: u32,
    pub order: Vec<usize>,
    pub position: usize,
    pub batch_size: usize,
    pub shuffle_state: Option<Vec<u8>>,
}

pub struct ResumableDataLoader<B: Backend, I, O> {
    dataset: Arc<dyn Dataset<I>>,
    batcher: Arc<dyn Batcher<B, I, O>>,
    batch_size: usize,
    workers: usize,
    reuse_workers: bool,
    worker_pool: Arc<Mutex<Option<ReadWorkers<I>>>>,
    batch_reading: bool,
    rng: Arc<Mutex<Option<CheckpointRng>>>,
    device: B::Device,
    context: Option<AutodiffContext>,
}

impl<B: Backend, I, O> Clone for ResumableDataLoader<B, I, O> {
    fn clone(&self) -> Self {
        Self {
            dataset: self.dataset.clone(), batcher: self.batcher.clone(),
            batch_size: self.batch_size, workers: self.workers, batch_reading: self.batch_reading,
            reuse_workers: self.reuse_workers, worker_pool: self.worker_pool.clone(),
            rng: self.rng.clone(), device: self.device.clone(), context: self.context,
        }
    }
}

impl<B: Backend, I, O> ResumableDataLoader<B, I, O> {
    pub fn new(dataset: Arc<dyn Dataset<I>>, batcher: Arc<dyn Batcher<B, I, O>>,
        device: B::Device, batch_size: usize, shuffle_seed: Option<u64>, workers: usize,
        batch_reading: bool) -> Self
    {
        assert!(batch_size > 0, "Batch size must be positive");
        Self { dataset, batcher, device, batch_size, workers, batch_reading,
            reuse_workers: false, worker_pool: Arc::new(Mutex::new(None)),
            rng: Arc::new(Mutex::new(shuffle_seed.map(CheckpointRng::seed_from_u64))), context: None }
    }

    pub fn reuse_workers(mut self, enabled: bool) -> Self {
        self.reuse_workers = enabled;
        if !enabled {
            self.worker_pool = Arc::new(Mutex::new(None));
        }
        self
    }

    fn initial_state(&self) -> DataLoaderState {
        let mut rng = self.rng.lock().unwrap();
        let mut order = (0..self.dataset.len()).collect::<Vec<_>>();
        if let Some(rng) = rng.as_mut() { order.shuffle(rng); }
        DataLoaderState {
            version: 1, order, position: 0, batch_size: self.batch_size,
            shuffle_state: rng.as_ref().map(checkpoint_rng),
        }
    }
}

struct ResumableIterator<B: Backend, I, O> {
    loader: ResumableDataLoader<B, I, O>,
    state: DataLoaderState,
    workers: Option<ReadWorkers<I>>,
}

type ReadJob<I> = Box<dyn FnOnce() -> Vec<I> + Send>;

struct ReadRequest<I> {
    read: ReadJob<I>,
    result: mpsc::SyncSender<thread::Result<Vec<I>>>,
}

struct ReadWorkers<I> {
    senders: Vec<mpsc::Sender<ReadRequest<I>>>,
    threads: Vec<thread::JoinHandle<()>>,
}

impl<I: Send + 'static> ReadWorkers<I> {
    fn new(count: usize) -> Self {
        let mut pool = Self { senders: Vec::with_capacity(count), threads: Vec::with_capacity(count) };
        for _ in 0..count {
            let (sender, receiver) = mpsc::channel::<ReadRequest<I>>();
            pool.senders.push(sender);
            pool.threads.push(thread::spawn(move || {
                while let Ok(request) = receiver.recv() {
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(request.read));
                    let failed = result.is_err();
                    if request.result.send(result).is_err() || failed { break; }
                }
            }));
        }
        pool
    }

    fn read(&self, jobs: Vec<ReadJob<I>>) -> thread::Result<Vec<I>> {
        let mut results = Vec::with_capacity(jobs.len());
        for (worker, read) in jobs.into_iter().enumerate() {
            let (result, receiver) = mpsc::sync_channel(1);
            self.senders[worker].send(ReadRequest { read, result })
                .map_err(|_| Box::new(String::from("resumable dataset worker stopped"))
                    as Box<dyn std::any::Any + Send>)?;
            results.push(receiver);
        }
        let mut items = Vec::new();
        for result in results {
            let batch = result.recv().map_err(|_| {
                Box::new(String::from("resumable dataset worker did not return a batch"))
                    as Box<dyn std::any::Any + Send>
            })??;
            items.extend(batch);
        }
        Ok(items)
    }
}

impl<I> Drop for ReadWorkers<I> {
    fn drop(&mut self) {
        self.senders.clear();
        for worker in self.threads.drain(..) {
            let _ = worker.join();
        }
    }
}

impl<B: Backend, I, O> Drop for ResumableIterator<B, I, O> {
    fn drop(&mut self) {
        if self.loader.reuse_workers && !thread::panicking() {
            let mut cached = self.loader.worker_pool.lock().unwrap_or_else(|error| error.into_inner());
            if cached.is_none() {
                *cached = self.workers.take();
            }
        }
    }
}

fn read_dataset_items<I>(dataset: &dyn Dataset<I>, indices: &[usize], batch_reading: bool) -> Vec<I> {
    if batch_reading {
        let items = dataset.get_many(indices).expect("resumable dataset item is missing");
        assert_eq!(items.len(), indices.len(), "resumable dataset returned the wrong number of items");
        items
    } else {
        indices.iter().map(|index| dataset.get(*index)
            .expect("resumable dataset item is missing")).collect()
    }
}

impl<B, I, O> ResumableIterator<B, I, O>
where B: Backend, I: Clone + Send + Sync + 'static, O: Send + 'static
{
    fn read_items(&mut self, indices: &[usize]) -> Vec<I> {
        if self.loader.workers == 0 || indices.is_empty() {
            return read_dataset_items(self.loader.dataset.as_ref(), indices, self.loader.batch_reading);
        }
        let count = if self.loader.dataset.is_read_order_independent() {
            self.loader.workers
        } else {
            1
        };
        if self.workers.is_none() {
            if self.loader.reuse_workers {
                self.workers = self.loader.worker_pool.lock()
                    .unwrap_or_else(|error| error.into_inner()).take();
            }
            if self.workers.as_ref().is_some_and(|pool| pool.senders.len() != count) {
                self.workers = None;
            }
            if self.workers.is_none() { self.workers = Some(ReadWorkers::new(count)); }
        }
        let chunk = indices.len().div_ceil(count.min(indices.len()));
        let jobs = indices.chunks(chunk).map(|indices| {
            let dataset = self.loader.dataset.clone();
            let device = self.loader.device.clone();
            let context = self.loader.context;
            let batch_reading = self.loader.batch_reading;
            let indices = indices.to_vec();
            Box::new(move || {
                let read = || read_dataset_items(dataset.as_ref(), &indices, batch_reading);
                match context {
                    Some(context) => B::with_autodiff_creation_context(&device, context, read),
                    None => read(),
                }
            }) as ReadJob<I>
        }).collect();
        match self.workers.as_ref().unwrap().read(jobs) {
            Ok(items) => items,
            Err(panic) => {
                drop(self.workers.take());
                std::panic::resume_unwind(panic)
            }
        }
    }
}

impl<B, I, O> Iterator for ResumableIterator<B, I, O>
where B: Backend, I: Clone + Send + Sync + 'static, O: Send + 'static
{
    type Item = O;
    fn next(&mut self) -> Option<O> {
        if self.state.position == self.state.order.len() { return None; }
        let end = self.state.position.saturating_add(self.state.batch_size).min(self.state.order.len());
        let indices = self.state.order[self.state.position..end].to_vec();
        let context = self.loader.context;
        let device = self.loader.device.clone();
        let mut make_batch = || {
            let items = self.read_items(&indices);
            self.loader.batcher.batch(items, &self.loader.device)
        };
        let output = match context {
            Some(context) => B::with_autodiff_creation_context(&device, context, make_batch),
            None => make_batch(),
        };
        self.state.position = end;
        Some(output)
    }
}

impl<B, I, O> DataLoaderIterator<O> for ResumableIterator<B, I, O>
where B: Backend, I: Clone + Send + Sync + 'static, O: Send + 'static
{
    fn progress(&self) -> Progress { Progress::new(self.state.position, self.state.order.len()) }
    fn checkpoint(&self) -> Result<DataLoaderState, String> { Ok(self.state.clone()) }
}

impl<B, I, O> DataLoader<B, O> for ResumableDataLoader<B, I, O>
where B: Backend, I: Clone + Send + Sync + 'static, O: Send + 'static
{
    fn checkpoint_source(&self) -> Result<Vec<u8>, String> {
        let state = (
            1u32, self.dataset.len(), self.batch_size, self.workers, self.batch_reading,
            self.rng.lock().unwrap().as_ref().map(checkpoint_rng),
            self.dataset.checkpoint()?, self.batcher.checkpoint()?,
        );
        bincode::serde::encode_to_vec(&state, bincode::config::standard()).map_err(|e| e.to_string())
    }

    fn restore_source(&self, bytes: &[u8]) -> Result<(), String> {
        type State = (u32, usize, usize, usize, bool, Option<Vec<u8>>, Vec<u8>, Vec<u8>);
        let (state, read): (State, _) = bincode::serde::decode_from_slice(bytes, bincode::config::standard())
            .map_err(|e| e.to_string())?;
        let (version, len, batch_size, workers, batch_reading, rng, dataset, batcher) = state;
        if read != bytes.len() || version != 1 || len != self.dataset.len()
            || batch_size != self.batch_size || workers != self.workers || batch_reading != self.batch_reading
            || rng.is_some() != self.rng.lock().unwrap().is_some()
        { return Err("Data source configuration differs from checkpoint".into()); }
        let rng = rng.as_deref().map(restore_rng).transpose().map_err(str::to_owned)?;
        self.dataset.restore_checkpoint(&dataset)?;
        self.batcher.restore_checkpoint(&batcher)?;
        *self.rng.lock().unwrap() = rng;
        Ok(())
    }
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<O> + 'a> {
        Box::new(ResumableIterator { loader: self.clone(), state: self.initial_state(), workers: None })
    }

    fn iter_from_checkpoint<'a>(&'a self, saved: Option<&DataLoaderState>)
        -> Result<Box<dyn DataLoaderIterator<O> + 'a>, String>
    {
        let state = match saved {
            None => self.initial_state(),
            Some(state) => {
                if state.version != 1 || state.order.len() != self.dataset.len()
                    || state.position > state.order.len() || state.batch_size != self.batch_size
                { return Err("Data loader checkpoint does not match the source or batch configuration".into()); }
                let mut seen = vec![false; state.order.len()];
                for index in &state.order {
                    let Some(present) = seen.get_mut(*index) else { return Err("Invalid shuffled dataset index".into()); };
                    if *present { return Err("Duplicate shuffled dataset index".into()); }
                    *present = true;
                }
                let rng = state.shuffle_state.as_deref().map(restore_rng).transpose()
                    .map_err(str::to_owned)?;
                if self.rng.lock().unwrap().is_some() != rng.is_some() {
                    return Err("Shuffle configuration differs from the checkpoint".into());
                }
                *self.rng.lock().unwrap() = rng;
                state.clone()
            }
        };
        Ok(Box::new(ResumableIterator { loader: self.clone(), state, workers: None }))
    }

    fn num_items(&self) -> usize { self.dataset.len() }
    fn to_device(&self, device: &B::Device) -> Arc<dyn DataLoader<B, O>> {
        let mut loader = self.clone();
        loader.device = device.clone();
        Arc::new(loader)
    }
    fn with_autodiff_context(&self, context: AutodiffContext) -> Arc<dyn DataLoader<B, O>> {
        let mut loader = self.clone();
        loader.context = Some(context);
        Arc::new(loader)
    }
    fn slice(&self, start: usize, end: usize) -> Arc<dyn DataLoader<B, O>> {
        let mut loader = self.clone();
        loader.dataset = Arc::new(PartialDataset::new(self.dataset.clone(), start, end));
        loader.worker_pool = Arc::new(Mutex::new(None));
        let state = self.rng.lock().unwrap().as_ref().map(checkpoint_rng);
        loader.rng = Arc::new(Mutex::new(state.as_deref().map(restore_rng).transpose().expect("valid RNG state")));
        Arc::new(loader)
    }
}
