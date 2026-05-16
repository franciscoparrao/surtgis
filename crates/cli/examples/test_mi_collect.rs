//! Validate the hypothesis that mi_collect(true) returns idle mimalloc
//! segments to the OS visibly (i.e. RSS drops in /proc/self/status), versus
//! a plain drop() which only returns pages to mimalloc's free list.
//!
//! Expected output if the hypothesis holds:
//!   baseline:                ~X MB
//!   after alloc 1 GB:        ~X+1000 MB
//!   after drop:              ~X+1000 MB        (unchanged — pages in free list)
//!   after mi_collect(true):  ~X MB             (back to baseline)

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn rss_mb() -> usize {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|kb| kb.parse::<usize>().ok())
                .map(|kb| kb / 1024)
        })
        .unwrap_or(0)
}

fn alloc_and_touch(n_mb: usize) -> Vec<u8> {
    let n = n_mb * 1024 * 1024;
    let mut v = vec![0u8; n];
    // Touch every page so it's actually faulted in and counts toward RSS.
    for i in (0..n).step_by(4096) {
        v[i] = (i & 0xff) as u8;
    }
    v
}

fn main() {
    println!("baseline:                  {} MB", rss_mb());

    let big = alloc_and_touch(1024); // 1 GB
    println!("after alloc 1 GB:          {} MB", rss_mb());

    drop(big);
    // Give mimalloc's purge/decay a moment if needed.
    std::thread::sleep(std::time::Duration::from_millis(100));
    println!("after drop:                {} MB", rss_mb());

    unsafe { libmimalloc_sys::mi_collect(true) };
    std::thread::sleep(std::time::Duration::from_millis(100));
    println!("after mi_collect(true):    {} MB", rss_mb());

    // Repeat to confirm the pattern is consistent across allocation cycles.
    println!();
    for round in 1..=3 {
        let big = alloc_and_touch(500);
        let after_alloc = rss_mb();
        drop(big);
        let after_drop = rss_mb();
        unsafe { libmimalloc_sys::mi_collect(true) };
        let after_collect = rss_mb();
        println!(
            "round {}: alloc 500 MB → {} MB | drop → {} MB | collect → {} MB",
            round, after_alloc, after_drop, after_collect,
        );
    }
}
