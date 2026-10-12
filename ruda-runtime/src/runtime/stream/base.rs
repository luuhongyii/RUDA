use alloc::vec::Vec;
use ruda_core::stream_id::StreamId;

/// Trait for creating streams, used by the stream pool to generate streams as needed.
pub trait StreamFactory {
    /// The type of stream produced by this factory.
    type Stream;
    /// Creates a new stream instance.
    fn create(&mut self) -> Self::Stream;
}

/// Represents a pool of streams, managing a collection of streams created by a factory.
#[derive(Debug)]
pub struct StreamPool<F: StreamFactory> {
    /// Vector storing optional streams, where None indicates an uninitialized stream.
    streams: Vec<Option<F::Stream>>,
    /// The factory used to create new streams when needed.
    factory: F,
    /// Maximum number of regular streams (excludes special streams).
    max_streams: usize,
}

impl<F: StreamFactory> StreamPool<F> {
    /// Creates a new stream pool with the given backend factory and capacity constraints.
    pub fn new(backend: F, max_streams: u8, num_special: u8) -> Self {
        // Initialize a vector with capacity for regular and special streams.
        let mut streams = Vec::with_capacity(max_streams as usize);
        // Pre-populate the vector with None to reserve space for all streams.
        for _ in 0..(max_streams.saturating_add(num_special)) {
            streams.push(None);
        }

        Self {
            streams,
            factory: backend,
            max_streams: max_streams as usize,
        }
    }

    /// Retrieves a mutable reference to a stream for a given stream ID.
    pub fn get_mut(&mut self, stream_id: &StreamId) -> &mut F::Stream {
        // Calculate the index for the stream ID.
        let index = self.stream_index(stream_id);

        self.get_mut_index(index)
    }

    /// Initialize an unused regular slot with a backend-specific stream. This
    /// does not modulo-map IDs or replace a stream that may have pending work.
    /// Returns false for an occupied slot or an ID outside the regular pool.
    pub fn try_initialize_with<E>(
        &mut self, stream_id: &StreamId,
        initialize: impl FnOnce(&mut F) -> Result<F::Stream, E>,
    ) -> Result<bool, E> {
        let Ok(index) = usize::try_from(stream_id.value) else { return Ok(false); };
        if index >= self.max_streams || self.streams[index].is_some() { return Ok(false); }
        let stream = initialize(&mut self.factory)?;
        self.streams[index] = Some(stream);
        Ok(true)
    }

    /// Retrieves a stream by index, initializing it if needed.
    pub fn get_mut_index(&mut self, index: usize) -> &mut F::Stream {
        let entry = &mut self.streams[index];
        entry.get_or_insert_with(|| self.factory.create())
    }

    /// Retrieves a special stream by index, initializing it if needed.
    pub fn get_special(&mut self, index: u8) -> &mut F::Stream {
        self.get_mut_index(self.max_streams + index as usize)
    }

    /// Calculates the index for a given stream ID, mapping it to the pool's capacity.
    pub fn stream_index(&mut self, id: &StreamId) -> usize {
        stream_index(id, self.max_streams)
    }
}

/// Maps a stream ID to an index within the pool's capacity using modulo arithmetic.
pub fn stream_index(stream_id: &StreamId, max_streams: usize) -> usize {
    stream_id.value as usize % max_streams
}
