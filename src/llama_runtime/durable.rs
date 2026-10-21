//! Durable file writes for the runtime-promotion path.
//!
//! A promotion journals its steps *after* the candidate set exists on disk, so
//! the candidate's files must already be durable when the journal is written:
//! otherwise a power loss can leave a journal that describes files whose
//! contents never reached the disk.

use std::fs::File;
use std::io::Write;
use std::path::Path;

fn synced(file: &File) -> std::io::Result<()> {
    file.sync_all()?;
    #[cfg(test)]
    SYNCS.with(|c| c.set(c.get() + 1));
    Ok(())
}

/// `fs::copy` that flushes the destination to disk before returning. The
/// destination is synced through the handle that wrote it, so a read-only
/// source mode (which `fs::copy` carries over) cannot block the flush.
pub fn copy_file(from: &Path, to: &Path) -> std::io::Result<u64> {
    let mut source = File::open(from)?;
    let mut dest = File::create(to)?;
    let copied = std::io::copy(&mut source, &mut dest)?;
    synced(&dest)?;
    drop(dest);
    std::fs::set_permissions(to, source.metadata()?.permissions())?;
    Ok(copied)
}

/// `fs::write` that flushes the file to disk before returning.
pub fn write_file(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    file.write_all(bytes)?;
    synced(&file)
}

/// Flush a directory's entries (new files, renames). A no-op where directory
/// handles cannot be synced (Windows).
pub fn sync_dir(dir: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        File::open(dir)?.sync_all()?;
    }
    #[cfg(not(unix))]
    {
        let _ = dir;
    }
    Ok(())
}

#[cfg(test)]
thread_local! {
    /// File syncs issued on this thread, so tests can prove the contract.
    pub(crate) static SYNCS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn syncs() -> usize {
    SYNCS.with(|c| c.get())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_file_copies_bytes_and_permissions_and_syncs_once() {
        let dir = tempfile::tempdir().unwrap();
        let from = dir.path().join("a");
        let to = dir.path().join("b");
        std::fs::write(&from, b"payload").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&from, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        let before = syncs();
        assert_eq!(copy_file(&from, &to).unwrap(), 7);
        assert_eq!(syncs() - before, 1);
        assert_eq!(std::fs::read(&to).unwrap(), b"payload");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&to).unwrap().permissions().mode() & 0o777,
                0o755
            );
        }
    }

    #[test]
    fn a_read_only_source_can_still_be_copied_and_synced() {
        let dir = tempfile::tempdir().unwrap();
        let from = dir.path().join("ro");
        std::fs::write(&from, b"x").unwrap();
        let mut perms = std::fs::metadata(&from).unwrap().permissions();
        perms.set_readonly(true);
        std::fs::set_permissions(&from, perms).unwrap();
        let before = syncs();
        copy_file(&from, &dir.path().join("copy")).unwrap();
        assert_eq!(syncs() - before, 1);
    }

    #[test]
    fn write_file_writes_and_syncs_once() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.json");
        let before = syncs();
        write_file(&path, b"{}").unwrap();
        assert_eq!(syncs() - before, 1);
        assert_eq!(std::fs::read(&path).unwrap(), b"{}");
    }

    #[test]
    fn sync_dir_accepts_a_real_directory() {
        sync_dir(tempfile::tempdir().unwrap().path()).unwrap();
    }
}
