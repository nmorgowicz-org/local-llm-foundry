//! Scratch: pure std TCP connect — discriminates reqwest/hyper from process-level blocking.
use std::net::TcpStream;

fn main() {
    match TcpStream::connect("192.168.2.16:8001") {
        Ok(_) => println!("RAW TCP: connected"),
        Err(e) => {
            eprintln!("RAW TCP ERR: {e} (os error {})", e.raw_os_error().unwrap_or(0));
            std::process::exit(1);
        }
    }
}
