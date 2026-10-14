use libloading::Library;
use std::{ffi::{c_char,c_void,CStr,CString},sync::{Arc,Mutex}};

#[derive(Debug)]
pub struct UcxError(String);
impl std::fmt::Display for UcxError {
    fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result {std::fmt::Display::fmt(&self.0,f)}
}
impl std::error::Error for UcxError {}
fn error(value:impl std::fmt::Display)->UcxError {UcxError(value.to_string())}

#[repr(C)]
struct Transfer {peer:u32,receive:u32,tag:u64,pointer:u64,bytes:usize}
type Create=unsafe extern "C" fn(u32,i32,*const c_char,*mut *mut c_void)->i32;
type Address=unsafe extern "C" fn(*mut c_void,*mut u8,*mut usize)->i32;
type Connect=unsafe extern "C" fn(*mut c_void,u32,*const u8)->i32;
type Exchange=unsafe extern "C" fn(*mut c_void,*const Transfer,usize,u64)->i32;
type Destroy=unsafe extern "C" fn(*mut c_void);
type LastError=unsafe extern "C" fn()->*const c_char;

pub struct UcxApi {
    _library:Library,create:Create,address:Address,connect:Connect,exchange:Exchange,destroy:Destroy,last_error:LastError,
}
impl UcxApi {
    pub fn load(path:&std::path::Path)->Result<Arc<Self>,UcxError> {
        // Retain the explicitly selected library for every worker and transfer.
        unsafe {
            let library=Library::new(path).map_err(error)?;
            let version=*library.get::<unsafe extern "C" fn()->u32>(b"ruccl_ucx_api_version\0").map_err(error)?;
            if version()!=1 {return Err(error("GPU KV needs ruCCL UCX C ABI version 1"));}
            Ok(Arc::new(Self{
                create:*library.get(b"ruccl_ucx_create\0").map_err(error)?,
                address:*library.get(b"ruccl_ucx_address\0").map_err(error)?,
                connect:*library.get(b"ruccl_ucx_connect\0").map_err(error)?,
                exchange:*library.get(b"ruccl_ucx_exchange\0").map_err(error)?,
                destroy:*library.get(b"ruccl_ucx_destroy\0").map_err(error)?,
                last_error:*library.get(b"ruccl_error\0").map_err(error)?,_library:library,
            }))
        }
    }
    fn status(&self,status:i32)->Result<(),UcxError> {
        if status==0 {return Ok(());}
        let message=unsafe {(self.last_error)()};
        if message.is_null() {Err(error(format!("ruCCL UCX status {status}")))}
        else {Err(error(unsafe {CStr::from_ptr(message)}.to_string_lossy()))}
    }
}

pub struct UcxEndpoint {api:Arc<UcxApi>,handle:Mutex<usize>,timeout_ms:u64}
impl UcxEndpoint {
    pub fn new(api:Arc<UcxApi>,device:i32,peers:usize,transports:&str,timeout_ms:u64)->Result<Self,UcxError> {
        let peers=u32::try_from(peers).map_err(error)?;
        if peers==0 || device<0 {return Err(error("GPU KV needs a real device and nonempty peer group"));}
        let transports=CString::new(transports).map_err(error)?;
        let mut handle=std::ptr::null_mut();
        api.status(unsafe {(api.create)(peers,device,transports.as_ptr(),&mut handle)})?;
        if handle.is_null() {return Err(error("ruCCL UCX returned a null worker"));}
        Ok(Self{api,handle:Mutex::new(handle as usize),timeout_ms})
    }
    pub fn address(&self)->Result<Vec<u8>,UcxError> {
        let handle=self.handle.lock().map_err(error)?;let mut length=0;
        self.api.status(unsafe {(self.api.address)(*handle as *mut c_void,std::ptr::null_mut(),&mut length)})?;
        let mut bytes=vec![0;length];
        self.api.status(unsafe {(self.api.address)(*handle as *mut c_void,bytes.as_mut_ptr(),&mut length)})?;
        if length!=bytes.len() {return Err(error("UCX address length changed"));}Ok(bytes)
    }
    pub fn connect(&self,peer:usize,address:&[u8])->Result<(),UcxError> {
        if address.is_empty() {return Err(error("empty UCX peer address"));}
        let peer=u32::try_from(peer).map_err(error)?;let handle=self.handle.lock().map_err(error)?;
        self.api.status(unsafe {(self.api.connect)(*handle as *mut c_void,peer,address.as_ptr())})
    }
    /// The bridge drains UCX operations, including timeout cancellation, before
    /// return. The caller owns and pins the GPU buffer throughout this call.
    pub fn transfer(&self,peer:usize,receive:bool,tag:u64,pointer:u64,bytes:usize)->Result<(),UcxError> {
        let transfer=Transfer{peer:u32::try_from(peer).map_err(error)?,receive:u32::from(receive),tag,pointer,bytes};
        let handle=self.handle.lock().map_err(error)?;
        self.api.status(unsafe {(self.api.exchange)(*handle as *mut c_void,&transfer,1,self.timeout_ms)})
    }
}
impl Drop for UcxEndpoint {
    fn drop(&mut self) {
        let handle=*self.handle.get_mut().unwrap_or_else(|e|e.into_inner());
        unsafe {(self.api.destroy)(handle as *mut c_void)};
    }
}
