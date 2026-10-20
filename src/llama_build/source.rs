//! The managed llama.cpp checkout (plan §4.4, decision D5).
//!
//! Only two refs are buildable: the latest `bNNNN` tag and `master`. Either is
//! resolved **once** to a full commit; that commit is then fetched by object id,
//! pinned with a job ref and checked out detached. Ref movement after
//! resolution (a new master commit, a re-pointed tag) cannot change what this
//! job builds.

use std::path::{Path, PathBuf};
use std::time::Duration;

use super::RefChoice;
use super::prereqs::LLAMA_CPP_REPO_URL;
use super::sysprobe::{CmdOutput, run_bounded};

/// Namespace for refs that keep a job's commit alive through GC.
pub const JOB_REF_PREFIX: &str = "refs/foundry/jobs/";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceError {
    Git { command: String, detail: String },
    NoTags,
    NoMaster,
    BadCommit(String),
    Mismatch { expected: String, found: String },
    Io(String),
}

impl std::fmt::Display for SourceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Git { command, detail } => write!(f, "`git {command}` failed: {detail}"),
            Self::NoTags => write!(f, "the repository has no bNNNN release tags"),
            Self::NoMaster => write!(f, "the repository has no master branch"),
            Self::BadCommit(c) => write!(f, "`{c}` is not a full commit id"),
            Self::Mismatch { expected, found } => {
                write!(f, "checked out {found} but {expected} was pinned")
            }
            Self::Io(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for SourceError {}

/// Runs git. The production implementation is bounded and non-interactive; the
/// job layer substitutes a supervised one so cancel reaches git too.
pub trait Git {
    fn run(&self, cwd: Option<&Path>, args: &[String]) -> Result<CmdOutput, String>;
}

pub struct RealGit {
    pub program: PathBuf,
    pub timeout: Duration,
    /// The remote this instance operates on, when known. Local-path remotes
    /// (test fixtures) are the only ones allowed to re-enable git's file
    /// transport; production remotes are https and keep git's safe default.
    remote: Option<String>,
}

impl RealGit {
    pub fn new(program: PathBuf) -> Self {
        Self {
            program,
            timeout: Duration::from_secs(600),
            remote: None,
        }
    }

    /// Record the remote so `protocol.file.allow` stays scoped to local paths
    /// (CVE-2022-39253 class stays closed for https remotes).
    pub fn with_remote(mut self, url: impl Into<String>) -> Self {
        self.remote = Some(url.into());
        self
    }
}

impl Git for RealGit {
    fn run(&self, cwd: Option<&Path>, args: &[String]) -> Result<CmdOutput, String> {
        let mut full: Vec<String> = Vec::new();
        if let Some(cwd) = cwd {
            full.push("-C".into());
            full.push(cwd.to_string_lossy().into_owned());
        }
        for c in [
            "advice.detachedHead=false",
            "gc.auto=0",
            "core.longpaths=true",
        ] {
            full.push("-c".into());
            full.push(c.into());
        }
        if allows_local_file_transport(self.remote.as_deref(), args) {
            full.push("-c".into());
            full.push("protocol.file.allow=always".into());
        }
        full.extend(args.iter().cloned());
        let refs: Vec<&str> = full.iter().map(String::as_str).collect();
        run_bounded(
            &self.program.to_string_lossy(),
            &refs,
            self.timeout,
            &[
                ("GIT_TERMINAL_PROMPT".into(), "0".into()),
                ("GIT_ASKPASS".into(), String::new()),
                ("LC_ALL".into(), "C".into()),
            ],
        )
    }
}

fn strs(args: &[&str]) -> Vec<String> {
    args.iter().map(|s| s.to_string()).collect()
}

fn git_ok(git: &dyn Git, cwd: Option<&Path>, args: &[&str]) -> Result<CmdOutput, SourceError> {
    let owned = strs(args);
    let out = git.run(cwd, &owned).map_err(|detail| SourceError::Git {
        command: args.join(" "),
        detail,
    })?;
    if out.ok() {
        Ok(out)
    } else {
        Err(SourceError::Git {
            command: args.join(" "),
            detail: out.combined().trim().chars().take(600).collect(),
        })
    }
}

pub fn is_full_commit(text: &str) -> bool {
    text.len() == 40 && text.bytes().all(|b| b.is_ascii_hexdigit())
}

// ---------------------------------------------------------------------------
// Resolution
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ResolvedRef {
    /// `b11727` or `master`.
    pub reference: String,
    pub commit: String,
    pub build_number: Option<u32>,
}

/// `bNNNN` → NNNN.
pub fn tag_build_number(tag: &str) -> Option<u32> {
    let digits = tag.strip_prefix('b')?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

/// Tags from `git ls-remote --tags`. A peeled `^{}` line (the commit an
/// annotated tag points at) wins over the tag object.
pub fn parse_ls_remote_tags(output: &str) -> Vec<(String, String)> {
    let mut tags: std::collections::BTreeMap<String, String> = Default::default();
    for line in output.lines() {
        let Some((sha, name)) = line.split_once('\t') else {
            continue;
        };
        let Some(tag) = name.trim().strip_prefix("refs/tags/") else {
            continue;
        };
        if !is_full_commit(sha.trim()) {
            continue;
        }
        match tag.strip_suffix("^{}") {
            Some(base) => {
                tags.insert(base.to_string(), sha.trim().to_string());
            }
            None => {
                tags.entry(tag.to_string())
                    .or_insert_with(|| sha.trim().to_string());
            }
        }
    }
    tags.into_iter().collect()
}

/// The highest `bNNNN` tag.
pub fn latest_tag(tags: &[(String, String)]) -> Option<ResolvedRef> {
    tags.iter()
        .filter_map(|(name, commit)| tag_build_number(name).map(|n| (n, name, commit)))
        .max_by_key(|(n, _, _)| *n)
        .map(|(n, name, commit)| ResolvedRef {
            reference: name.clone(),
            commit: commit.clone(),
            build_number: Some(n),
        })
}

pub fn parse_ls_remote_head(output: &str) -> Option<String> {
    output
        .lines()
        .filter_map(|l| l.split_once('\t'))
        .find(|(_, name)| name.trim() == "refs/heads/master")
        .map(|(sha, _)| sha.trim().to_string())
        .filter(|sha| is_full_commit(sha))
}

/// Resolve the chosen ref against `url` to an exact commit.
pub fn resolve(git: &dyn Git, url: &str, choice: RefChoice) -> Result<ResolvedRef, SourceError> {
    match choice {
        RefChoice::LatestTag => {
            let out = git_ok(git, None, &["ls-remote", "--tags", url, "refs/tags/b*"])?;
            latest_tag(&parse_ls_remote_tags(&out.stdout)).ok_or(SourceError::NoTags)
        }
        RefChoice::Master => {
            let out = git_ok(git, None, &["ls-remote", url, "refs/heads/master"])?;
            let commit = parse_ls_remote_head(&out.stdout).ok_or(SourceError::NoMaster)?;
            // Upstream tags each master commit; reuse that for the build
            // number when the tag already exists.
            let tags = git_ok(git, None, &["ls-remote", "--tags", url, "refs/tags/b*"])
                .map(|o| parse_ls_remote_tags(&o.stdout))
                .unwrap_or_default();
            let build_number = tags
                .iter()
                .filter(|(_, c)| *c == commit)
                .filter_map(|(n, _)| tag_build_number(n))
                .max();
            Ok(ResolvedRef {
                reference: "master".into(),
                commit,
                build_number,
            })
        }
    }
}

/// `b11727-3-g098ce52` (from `git describe --tags --long`) → 11730, the build
/// number llama.cpp itself reports for a commit that many past the tag.
pub fn normalize_describe(describe: &str) -> Option<u32> {
    let describe = describe.trim();
    let (tag_and_count, hash) = describe.rsplit_once("-g")?;
    if hash.is_empty() || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let (tag, count) = tag_and_count.rsplit_once('-')?;
    let base = tag_build_number(tag)?;
    let ahead: u32 = count.parse().ok()?;
    base.checked_add(ahead)
}

/// Whether a runtime's reported identity proves it was built from `pinned`.
/// A matching build **number** alone never counts: a different commit can
/// report the same number (a fork, a re-tag, a local patch).
pub fn commit_proves_pin(pinned: &str, reported_commit: Option<&str>) -> bool {
    is_full_commit(pinned)
        && reported_commit.is_some_and(|short| {
            short.len() >= 7
                && short.bytes().all(|b| b.is_ascii_hexdigit())
                && pinned
                    .to_ascii_lowercase()
                    .starts_with(&short.to_ascii_lowercase())
        })
}

// ---------------------------------------------------------------------------
// Checkout
// ---------------------------------------------------------------------------

fn job_ref(job_id: &str) -> Result<String, SourceError> {
    if job_id.is_empty()
        || !job_id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        return Err(SourceError::Io(format!("`{job_id}` is not a valid job id")));
    }
    Ok(format!("{JOB_REF_PREFIX}{job_id}"))
}

/// Make `src` a checkout of `url` at exactly `commit`.
///
/// Fetches only that object, shallowly. The commit is pinned by a job ref
/// before the working tree is touched, so garbage collection cannot remove it
/// while the job runs. Untracked and ignored files are removed: build trees
/// live outside `src/`, so nothing of value is lost.
pub fn checkout_exact(
    git: &dyn Git,
    src: &Path,
    url: &str,
    commit: &str,
    job_id: &str,
) -> Result<(), SourceError> {
    if !is_full_commit(commit) {
        return Err(SourceError::BadCommit(commit.to_string()));
    }
    let pin = job_ref(job_id)?;
    std::fs::create_dir_all(src).map_err(|e| SourceError::Io(e.to_string()))?;
    if !src.join(".git").exists() {
        git_ok(git, Some(src), &["init", "--quiet"])?;
    }
    // The remote is always ours; reset it rather than trust a stored URL.
    if git_ok(git, Some(src), &["remote", "set-url", "origin", url]).is_err() {
        git_ok(git, Some(src), &["remote", "add", "origin", url])?;
    }
    git_ok(
        git,
        Some(src),
        &[
            "fetch",
            "--quiet",
            "--depth",
            "1",
            "--no-tags",
            "origin",
            commit,
        ],
    )?;
    git_ok(git, Some(src), &["update-ref", &pin, commit])?;
    git_ok(
        git,
        Some(src),
        &["checkout", "--quiet", "--detach", "--force", commit],
    )?;
    git_ok(git, Some(src), &["reset", "--quiet", "--hard", commit])?;
    git_ok(git, Some(src), &["clean", "--quiet", "-fdx"])?;
    let head = git_ok(git, Some(src), &["rev-parse", "HEAD"])?;
    let found = head.stdout.trim().to_string();
    if found != commit {
        return Err(SourceError::Mismatch {
            expected: commit.to_string(),
            found,
        });
    }
    Ok(())
}

/// Drop a finished job's pin.
pub fn unpin(git: &dyn Git, src: &Path, job_id: &str) -> Result<(), SourceError> {
    let pin = job_ref(job_id)?;
    // Deleting an absent ref is not an error worth reporting.
    let _ = git_ok(git, Some(src), &["update-ref", "-d", &pin]);
    Ok(())
}

pub fn pinned_jobs(git: &dyn Git, src: &Path) -> Result<Vec<String>, SourceError> {
    let out = git_ok(
        git,
        Some(src),
        &["for-each-ref", "--format=%(refname)", JOB_REF_PREFIX],
    )?;
    Ok(out
        .stdout
        .lines()
        .filter_map(|l| l.trim().strip_prefix(JOB_REF_PREFIX))
        .map(str::to_string)
        .collect())
}

/// Bounded retention: remove pins for jobs that are no longer active, then
/// collect unreachable objects. Pins of active jobs survive, so a concurrent
/// reader's commit cannot disappear.
pub fn prune_inactive(
    git: &dyn Git,
    src: &Path,
    is_active: &dyn Fn(&str) -> bool,
) -> Result<Vec<String>, SourceError> {
    if !src.join(".git").exists() {
        return Ok(Vec::new());
    }
    let mut removed = Vec::new();
    for job in pinned_jobs(git, src)? {
        if !is_active(&job) {
            unpin(git, src, &job)?;
            removed.push(job);
        }
    }
    let _ = git_ok(
        git,
        Some(src),
        &["reflog", "expire", "--expire=now", "--all"],
    );
    let _ = git_ok(git, Some(src), &["gc", "--quiet", "--prune=now"]);
    Ok(removed)
}

pub fn head_commit(git: &dyn Git, src: &Path) -> Result<String, SourceError> {
    Ok(git_ok(git, Some(src), &["rev-parse", "HEAD"])?
        .stdout
        .trim()
        .to_string())
}

pub fn default_remote() -> &'static str {
    LLAMA_CPP_REPO_URL
}

/// `protocol.file.allow=always` re-opens the indirect `file://` transport
/// class git closed after CVE-2022-39253. It is only acceptable while every
/// remote involved is a local path (test fixtures); https remotes keep git's
/// safe default. A remote URL counts as local when it carries no network
/// scheme, and an argument counts when it is itself a local path or `file://`
/// URL (`fetch` by remote name hides the URL, so the remote must be known).
pub(crate) fn allows_local_file_transport(remote: Option<&str>, args: &[String]) -> bool {
    remote.is_some_and(is_local_remote_url) || args.iter().any(|a| is_local_transport_arg(a))
}

fn is_local_remote_url(url: &str) -> bool {
    const NETWORK_SCHEMES: [&str; 6] = [
        "https://", "http://", "ssh://", "git://", "ftp://", "ftps://",
    ];
    !NETWORK_SCHEMES.iter().any(|scheme| url.starts_with(scheme)) && !url.starts_with("git@")
}

fn is_local_transport_arg(arg: &str) -> bool {
    let windows_drive = {
        let bytes = arg.as_bytes();
        bytes.len() >= 3
            && bytes[0].is_ascii_alphabetic()
            && bytes[1] == b':'
            && (bytes[2] == b'\\' || bytes[2] == b'/')
    };
    arg.starts_with('/')
        || arg.starts_with("./")
        || arg.starts_with("../")
        || arg.starts_with("file://")
        || windows_drive
}

#[cfg(test)]
mod tests {
    use super::*;

    fn real_git() -> Option<RealGit> {
        which::which("git").ok().map(RealGit::new)
    }

    fn sh(git: &RealGit, cwd: &Path, args: &[&str]) -> String {
        let owned = strs(args);
        let out = git.run(Some(cwd), &owned).unwrap();
        assert!(out.ok(), "git {args:?}: {}", out.combined());
        out.stdout.trim().to_string()
    }

    /// A "remote" with `b1`, `b2` (annotated), and master two past b2.
    struct Remote {
        _dir: tempfile::TempDir,
        bare: PathBuf,
        work: PathBuf,
        git: RealGit,
        b1: String,
        b2: String,
    }

    impl Remote {
        fn url(&self) -> String {
            self.bare.to_string_lossy().into_owned()
        }

        fn commit(&self, file: &str, body: &str) -> String {
            std::fs::write(self.work.join(file), body).unwrap();
            sh(&self.git, &self.work, &["add", "-A"]);
            sh(
                &self.git,
                &self.work,
                &[
                    "-c",
                    "user.name=t",
                    "-c",
                    "user.email=t@t",
                    "commit",
                    "-q",
                    "-m",
                    body,
                ],
            );
            sh(&self.git, &self.work, &["rev-parse", "HEAD"])
        }

        fn push(&self) {
            sh(
                &self.git,
                &self.work,
                &["push", "-q", "-f", "origin", "master", "--tags", "--force"],
            );
        }
    }

    fn remote() -> Option<Remote> {
        let dir = tempfile::tempdir().unwrap();
        let bare = dir.path().join("remote.git");
        let work = dir.path().join("work");
        // The fixture remote is a local path; recording it keeps the
        // file-transport allowance scoped to fixtures only.
        let git = real_git()?.with_remote(bare.to_string_lossy().into_owned());
        std::fs::create_dir_all(&bare).unwrap();
        std::fs::create_dir_all(&work).unwrap();
        sh(
            &git,
            &bare,
            &["init", "-q", "--bare", "--initial-branch=master"],
        );
        // GitHub serves any reachable object id; mirror that locally.
        sh(
            &git,
            &bare,
            &["config", "uploadpack.allowAnySHA1InWant", "true"],
        );
        sh(&git, &work, &["init", "-q", "--initial-branch=master"]);
        sh(
            &git,
            &work,
            &["remote", "add", "origin", &bare.to_string_lossy()],
        );
        let r = Remote {
            _dir: dir,
            bare,
            work,
            git,
            b1: String::new(),
            b2: String::new(),
        };
        let b1 = r.commit("f.txt", "one");
        sh(&r.git, &r.work, &["tag", "b1"]);
        let b2 = r.commit("f.txt", "two");
        sh(
            &r.git,
            &r.work,
            &[
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@t",
                "tag",
                "-a",
                "b2",
                "-m",
                "release",
            ],
        );
        r.commit("f.txt", "three");
        r.commit("f.txt", "four");
        sh(&r.git, &r.work, &["tag", "not-a-build"]);
        r.push();
        Some(Remote { b1, b2, ..r })
    }

    #[test]
    fn tag_numbers() {
        assert_eq!(tag_build_number("b11727"), Some(11727));
        for bad in ["b", "11727", "b11727x", "bb1", "v1.0", "b-1", ""] {
            assert_eq!(tag_build_number(bad), None, "{bad}");
        }
    }

    #[test]
    fn ls_remote_parsing_prefers_peeled_commits_and_picks_the_highest_tag() {
        let sha = |c: char| c.to_string().repeat(40);
        let out = format!(
            "{}\trefs/tags/b9\n{}\trefs/tags/b10\n{}\trefs/tags/b10^{{}}\n{}\trefs/tags/b100\nbad line\n{}\trefs/tags/release\n{}\trefs/heads/x\n",
            sha('1'),
            sha('2'),
            sha('3'),
            sha('4'),
            sha('5'),
            sha('6'),
        );
        let tags = parse_ls_remote_tags(&out);
        assert!(tags.contains(&("b10".to_string(), sha('3'))), "peeled wins");
        let latest = latest_tag(&tags).unwrap();
        assert_eq!(latest.reference, "b100");
        assert_eq!(latest.build_number, Some(100));
        assert_eq!(latest.commit, sha('4'));
        assert!(latest_tag(&parse_ls_remote_tags("")).is_none());
    }

    #[test]
    fn describe_is_normalized_to_a_build_number() {
        assert_eq!(normalize_describe("b11727-3-g098ce52"), Some(11730));
        assert_eq!(normalize_describe("b11727-0-g098ce526b\n"), Some(11727));
        for bad in [
            "b11727",
            "v1-2-gabc",
            "b1-x-gabc",
            "b1-2-gzzz",
            "b1-2-g",
            "",
        ] {
            assert_eq!(normalize_describe(bad), None, "{bad}");
        }
    }

    #[test]
    fn a_build_number_alone_never_proves_the_pin() {
        let pinned = "098ce526b0000000000000000000000000000000";
        assert!(commit_proves_pin(pinned, Some("098ce52")));
        assert!(commit_proves_pin(pinned, Some("098CE526B")));
        // Same build number, different commit, or no commit at all.
        assert!(!commit_proves_pin(pinned, Some("deadbee")));
        assert!(!commit_proves_pin(pinned, None));
        assert!(!commit_proves_pin(pinned, Some("098ce")), "too short");
        assert!(
            !commit_proves_pin("b11727", Some("b11727")),
            "pin must be a full commit"
        );
        assert!(!commit_proves_pin(pinned, Some("")));
    }

    #[test]
    fn resolves_the_latest_tag_and_master() {
        let Some(r) = remote() else { return };
        let tag = resolve(&r.git, &r.url(), RefChoice::LatestTag).unwrap();
        assert_eq!(tag.reference, "b2");
        assert_eq!(
            tag.commit, r.b2,
            "an annotated tag resolves to its commit, not the tag object"
        );
        assert_eq!(tag.build_number, Some(2));
        let master = resolve(&r.git, &r.url(), RefChoice::Master).unwrap();
        assert_eq!(master.reference, "master");
        assert_eq!(master.commit, sh(&r.git, &r.work, &["rev-parse", "master"]));
        assert_ne!(master.commit, tag.commit);
        assert_eq!(master.build_number, None, "master is not tagged here");
    }

    #[test]
    fn a_tagged_master_reuses_the_tag_number() {
        let Some(r) = remote() else { return };
        sh(&r.git, &r.work, &["tag", "b4"]);
        r.push();
        let master = resolve(&r.git, &r.url(), RefChoice::Master).unwrap();
        assert_eq!(master.build_number, Some(4));
        assert_eq!(
            resolve(&r.git, &r.url(), RefChoice::LatestTag)
                .unwrap()
                .commit,
            master.commit
        );
    }

    #[test]
    fn an_unreachable_remote_is_a_git_error_with_the_command() {
        let Some(git) = real_git() else { return };
        let err = resolve(&git, "/definitely/not/a/repo", RefChoice::LatestTag).unwrap_err();
        assert!(
            matches!(&err, SourceError::Git { command, .. } if command.starts_with("ls-remote")),
            "{err:?}"
        );
        let empty = tempfile::tempdir().unwrap();
        sh(&git, empty.path(), &["init", "-q", "--bare"]);
        assert_eq!(
            resolve(&git, &empty.path().to_string_lossy(), RefChoice::LatestTag).unwrap_err(),
            SourceError::NoTags
        );
        assert_eq!(
            resolve(&git, &empty.path().to_string_lossy(), RefChoice::Master).unwrap_err(),
            SourceError::NoMaster
        );
    }

    #[test]
    fn master_and_tag_movement_after_resolution_cannot_change_the_job() {
        let Some(r) = remote() else { return };
        let resolved = resolve(&r.git, &r.url(), RefChoice::Master).unwrap();
        // Upstream moves on, and a tag is force-moved, after resolution.
        let newer = r.commit("f.txt", "five");
        sh(&r.git, &r.work, &["tag", "-f", "b2"]);
        r.push();
        assert_ne!(newer, resolved.commit);

        let src = tempfile::tempdir().unwrap();
        checkout_exact(&r.git, src.path(), &r.url(), &resolved.commit, "job1").unwrap();
        assert_eq!(head_commit(&r.git, src.path()).unwrap(), resolved.commit);
        assert_eq!(
            std::fs::read_to_string(src.path().join("f.txt")).unwrap(),
            "four"
        );
    }

    #[test]
    fn checkout_discards_local_changes_and_untracked_files_only_inside_src() {
        let Some(r) = remote() else { return };
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        let build_tree = tmp.path().join("build-metal");
        std::fs::create_dir_all(&build_tree).unwrap();
        std::fs::write(build_tree.join("keep.o"), b"object").unwrap();
        checkout_exact(&r.git, &src, &r.url(), &r.b2, "job1").unwrap();
        std::fs::write(src.join("f.txt"), "locally edited").unwrap();
        std::fs::write(src.join("stray.bin"), "junk").unwrap();
        checkout_exact(&r.git, &src, &r.url(), &r.b1, "job2").unwrap();
        assert_eq!(std::fs::read_to_string(src.join("f.txt")).unwrap(), "one");
        assert!(!src.join("stray.bin").exists());
        assert!(build_tree.join("keep.o").exists());
        assert_eq!(head_commit(&r.git, &src).unwrap(), r.b1);
    }

    #[test]
    fn the_job_pin_keeps_the_commit_through_gc_until_released() {
        let Some(r) = remote() else { return };
        let src = tempfile::tempdir().unwrap();
        checkout_exact(&r.git, src.path(), &r.url(), &r.b2, "active").unwrap();
        // A second job moves HEAD elsewhere; the first job's commit is only
        // kept alive by its pin.
        checkout_exact(&r.git, src.path(), &r.url(), &r.b1, "other").unwrap();
        assert_eq!(pinned_jobs(&r.git, src.path()).unwrap().len(), 2);
        let removed = prune_inactive(&r.git, src.path(), &|id| id == "active").unwrap();
        assert_eq!(removed, vec!["other".to_string()]);
        let exists = |c: &str| {
            r.git
                .run(
                    Some(src.path()),
                    &strs(&["cat-file", "-e", &format!("{c}^{{commit}}")]),
                )
                .unwrap()
                .ok()
        };
        assert!(exists(&r.b2), "the active job's commit survived gc");
        unpin(&r.git, src.path(), "active").unwrap();
        assert!(pinned_jobs(&r.git, src.path()).unwrap().is_empty());
    }

    #[test]
    fn malformed_commits_and_job_ids_are_rejected_before_any_git_call() {
        struct Never;
        impl Git for Never {
            fn run(&self, _: Option<&Path>, _: &[String]) -> Result<CmdOutput, String> {
                panic!("git must not be called");
            }
        }
        let dir = tempfile::tempdir().unwrap();
        for bad in [
            "master",
            "abc123",
            &"z".repeat(40),
            &"a".repeat(39),
            "--upload-pack=evil",
        ] {
            assert!(
                matches!(
                    checkout_exact(&Never, dir.path(), "u", bad, "j"),
                    Err(SourceError::BadCommit(_))
                ),
                "{bad}"
            );
        }
        let ok = "a".repeat(40);
        for bad in ["", "../x", "a b", "a/b", "x;y"] {
            assert!(
                checkout_exact(&Never, dir.path(), "u", &ok, bad).is_err(),
                "{bad}"
            );
        }
    }

    #[test]
    fn a_failed_fetch_surfaces_the_git_error() {
        let Some(git) = real_git() else { return };
        let src = tempfile::tempdir().unwrap();
        let err =
            checkout_exact(&git, src.path(), "/no/such/remote", &"a".repeat(40), "j").unwrap_err();
        assert!(
            matches!(err, SourceError::Git { ref command, .. } if command.starts_with("fetch")),
            "{err:?}"
        );
    }

    #[test]
    fn file_transport_allowance_stays_scoped_to_local_remotes() {
        assert!(is_local_remote_url("/tmp/fixture.git"));
        assert!(is_local_remote_url("file:///tmp/fixture.git"));
        assert!(!is_local_remote_url(
            "https://github.com/ggml-org/llama.cpp"
        ));
        assert!(!is_local_remote_url(
            "git@github.com:ggml-org/llama.cpp.git"
        ));
        assert!(!is_local_remote_url(
            "ssh://git@github.com/ggml-org/llama.cpp.git"
        ));
        // A bare `fetch origin <sha>` hides the URL; only remote knowledge
        // can enable the flag.
        assert!(allows_local_file_transport(
            Some("/tmp/fixture.git"),
            &[
                "fetch".into(),
                "--quiet".into(),
                "origin".into(),
                "abc".into()
            ]
        ));
        assert!(!allows_local_file_transport(
            Some("https://github.com/ggml-org/llama.cpp"),
            &[
                "fetch".into(),
                "--quiet".into(),
                "origin".into(),
                "abc".into()
            ]
        ));
        // A local path given directly in argv also enables it.
        assert!(allows_local_file_transport(
            None,
            &["ls-remote".into(), "/tmp/fixture.git".into()]
        ));
        assert!(!allows_local_file_transport(
            None,
            &["update-ref".into(), "refs/jobs/j1".into(), "abc123".into()]
        ));
    }

    #[test]
    fn prune_on_a_missing_checkout_is_a_no_op() {
        let Some(git) = real_git() else { return };
        let dir = tempfile::tempdir().unwrap();
        assert!(
            prune_inactive(&git, &dir.path().join("none"), &|_| false)
                .unwrap()
                .is_empty()
        );
    }
}
