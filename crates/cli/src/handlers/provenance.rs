//! `surtgis provenance` and `surtgis verify`: read the record embedded in
//! a SurtGIS output and check it against the file and its inputs.

use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};

use surtgis_core::Raster;
use surtgis_core::dispatch_any;
use surtgis_core::io::{raster_data_hash, read_geotiff_any, read_geotiff_bands, read_provenance};
use surtgis_core::provenance::{Provenance, hash_file, redact_source};

/// Records written by 1.5.0–1.6.0 stored signed URLs (Planetary Computer
/// SAS tokens) verbatim; never print them back.
fn redacted(mut p: Provenance) -> Provenance {
    for i in &mut p.inputs {
        i.source = redact_source(&i.source);
    }
    p
}

/// `surtgis provenance <file> [--json | --stac]`.
pub fn show(input: PathBuf, json: bool, stac: bool) -> Result<()> {
    let Some(p) = read_provenance(&input).context("Failed to read provenance")? else {
        println!("{}: no provenance record", input.display());
        return Ok(());
    };
    let p = redacted(p);
    if stac {
        println!(
            "{}",
            serde_json::to_string_pretty(&p.to_stac_processing()).expect("serialisable")
        );
        return Ok(());
    }
    if json {
        println!("{}", p.to_json_pretty());
        return Ok(());
    }
    println!("File: {}", input.display());
    println!(
        "Engine: {} {}  ({})",
        p.engine.name, p.engine.version, p.platform
    );
    println!("Created: {}", p.created);
    if let Some(op) = &p.operation {
        println!("Operation: {op}");
    }
    if !p.command.is_empty() {
        println!("Command: {}", p.command.join(" "));
    }
    if let Some(cwd) = &p.cwd {
        println!("Working dir: {cwd}");
    }
    if let Some(t) = p.threads {
        println!("Threads: {t}");
    }
    if let Some(s) = p.seed {
        println!("Seed: {s}");
    }
    if !p.parameters.is_null() {
        println!("Parameters: {}", p.parameters);
    }
    println!("Inputs ({}):", p.inputs.len());
    for i in &p.inputs {
        let digest = i
            .blake3
            .as_deref()
            .map(|h| format!("blake3:{}…", &h[..16.min(h.len())]))
            .unwrap_or_else(|| "(not hashed)".into());
        let size = i.bytes.map(|b| format!(", {b} bytes")).unwrap_or_default();
        let role = i
            .role
            .as_deref()
            .map(|r| format!(" [{r}]"))
            .unwrap_or_default();
        println!("  {}{}  {}{}", i.source, role, digest, size);
    }
    if !p.stac_items.is_empty() {
        println!("STAC items ({}):", p.stac_items.len());
        for it in &p.stac_items {
            println!(
                "  {}/{}  [{}]  accessed {}",
                it.collection,
                it.id,
                it.assets.join(","),
                it.accessed
            );
        }
    }
    match &p.output {
        Some(o) => println!(
            "Output: {} {}x{}, {} band(s), blake3:{}",
            o.dtype, o.shape[0], o.shape[1], o.bands, o.data_blake3
        ),
        None => println!("Output: (no digest)"),
    }
    Ok(())
}

/// `verify --against`: a list of affected STAC item ids and an optional
/// window on the time each item was accessed.
pub struct AffectedList {
    /// One id (or item URL) per line.
    pub file: PathBuf,
    /// Lower bound on `accessed`, inclusive.
    pub from: Option<String>,
    /// Upper bound on `accessed`, inclusive.
    pub to: Option<String>,
}

/// Item ids in a list file: blank lines and `#` comments skipped; an item
/// URL (`…/collections/c/items/<id>`) or `collection/id` reduces to `<id>`.
fn read_affected_ids(path: &Path) -> Result<std::collections::HashSet<String>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read {}", path.display()))?;
    Ok(text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            let l = l
                .split(['?', '#'])
                .next()
                .unwrap_or(l)
                .trim_end_matches('/');
            l.rsplit('/').next().unwrap_or(l).to_string()
        })
        .collect())
}

/// Seconds since the Unix epoch for `YYYY-MM-DD[THH:MM[:SS[.fff]]][Z|±HH:MM]`.
/// A bare date is midnight, or the end of the day when `end_of_day`.
fn parse_utc(s: &str, end_of_day: bool) -> Result<i64> {
    let bad = || anyhow::anyhow!("'{s}' is not an RFC 3339 time or YYYY-MM-DD date");
    let s = s.trim();
    let (date, time) = match s.split_once(['T', ' ']) {
        Some((d, t)) => (d, Some(t)),
        None => (s, None),
    };
    let mut d = date.split('-').map(|x| x.parse::<i64>());
    let (y, m, day) = match (d.next(), d.next(), d.next(), d.next()) {
        (Some(Ok(y)), Some(Ok(m)), Some(Ok(day)), None)
            if (1..=12).contains(&m) && (1..=31).contains(&day) =>
        {
            (y, m, day)
        }
        _ => return Err(bad()),
    };
    // Days from civil (Howard Hinnant's algorithm).
    let yy = if m <= 2 { y - 1 } else { y };
    let era = yy.div_euclid(400);
    let yoe = yy - era * 400;
    let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    let Some(t) = time else {
        return Ok(days * 86_400 + if end_of_day { 86_399 } else { 0 });
    };
    let (clock, offset) = if let Some(c) = t.strip_suffix(['Z', 'z']) {
        (c, 0)
    } else if let Some(i) = t.rfind(['+', '-']).filter(|&i| i >= 5) {
        let (c, o) = t.split_at(i);
        let sign = if o.starts_with('-') { -1 } else { 1 };
        let mut hm = o[1..].split(':').map(|x| x.parse::<i64>());
        match (hm.next(), hm.next()) {
            (Some(Ok(h)), Some(Ok(mi))) => (c, sign * (h * 3600 + mi * 60)),
            _ => return Err(bad()),
        }
    } else {
        (t, 0)
    };
    let mut c = clock.split(':');
    let h: i64 = c.next().and_then(|x| x.parse().ok()).ok_or_else(bad)?;
    let mi: i64 = c.next().and_then(|x| x.parse().ok()).ok_or_else(bad)?;
    let sec: i64 = match c.next() {
        Some(x) => x
            .split('.')
            .next()
            .and_then(|x| x.parse().ok())
            .ok_or_else(bad)?,
        None => 0,
    };
    Ok(days * 86_400 + h * 3600 + mi * 60 + sec - offset)
}

/// Items of `p` on the list and inside the window, as `(collection/id,
/// accessed)`, plus how many were on the list but accessed outside it.
fn affected_items(p: &Provenance, list: &AffectedList) -> Result<(Vec<(String, String)>, usize)> {
    let ids = read_affected_ids(&list.file)?;
    let from = list
        .from
        .as_deref()
        .map(|t| parse_utc(t, false))
        .transpose()?;
    let to = list.to.as_deref().map(|t| parse_utc(t, true)).transpose()?;
    let mut hits = Vec::new();
    let mut outside = 0usize;
    for it in &p.stac_items {
        if !ids.contains(&it.id) {
            continue;
        }
        let at = parse_utc(&it.accessed, false)?;
        if from.is_some_and(|f| at < f) || to.is_some_and(|t| at > t) {
            outside += 1;
            continue;
        }
        hits.push((format!("{}/{}", it.collection, it.id), it.accessed.clone()));
    }
    Ok((hits, outside))
}

/// `surtgis verify <file> [--skip-inputs] [--against <list>]`. Exit status
/// is non-zero when the output digest or any hashed input does not match,
/// or when the output read an item of the `--against` list.
pub fn verify(input: PathBuf, skip_inputs: bool, against: Option<AffectedList>) -> Result<()> {
    let Some(p) = read_provenance(&input).context("Failed to read provenance")? else {
        bail!(
            "{}: no provenance record to verify against",
            input.display()
        );
    };
    let p = redacted(p);
    let Some(expected) = &p.output else {
        bail!("{}: record carries no output digest", input.display());
    };

    let actual = output_digest(&input, &expected.dtype, expected.bands)?;
    let output_ok = actual == expected.data_blake3;
    println!(
        "Output  {}  {}",
        if output_ok { "OK      " } else { "MISMATCH" },
        input.display()
    );
    if !output_ok {
        println!("  recorded blake3:{}", expected.data_blake3);
        println!("  actual   blake3:{actual}");
    }

    let mut failed = !output_ok;
    let mut unverifiable = 0usize;
    if !skip_inputs {
        for i in &p.inputs {
            let Some(recorded) = &i.blake3 else {
                println!("Input   skipped   {} (not hashed at write time)", i.source);
                unverifiable += 1;
                continue;
            };
            let Some(path) = locate(&i.source, p.cwd.as_deref()) else {
                println!("Input   missing   {}", i.source);
                unverifiable += 1;
                continue;
            };
            let (digest, _) =
                hash_file(&path).with_context(|| format!("hashing {}", path.display()))?;
            let ok = &digest == recorded;
            println!(
                "Input   {}  {}",
                if ok { "OK      " } else { "MISMATCH" },
                path.display()
            );
            failed |= !ok;
        }
    }

    let mut affected = 0usize;
    if let Some(list) = &against {
        if p.stac_items.is_empty() {
            let remote = p.inputs.iter().any(|i| i.source.contains("://"));
            println!(
                "Items   unknown   no STAC item record{}",
                if remote {
                    " (written before SurtGIS 1.7, or not from a STAC composite); \
                     cannot check against the list"
                } else {
                    ""
                }
            );
            unverifiable += 1;
        } else {
            let (hits, outside) = affected_items(&p, list)?;
            for (item, at) in &hits {
                println!("Item    AFFECTED  {item} (accessed {at})");
            }
            if hits.is_empty() {
                if outside > 0 {
                    println!(
                        "Items   OK        {outside} of {} STAC item(s) on {} but accessed \
                         outside the affected window",
                        p.stac_items.len(),
                        list.file.display()
                    );
                } else {
                    println!(
                        "Items   OK        none of {} STAC item(s) is on {}",
                        p.stac_items.len(),
                        list.file.display()
                    );
                }
            }
            affected = hits.len();
        }
    }

    if affected > 0 {
        bail!(
            "{affected} STAC item(s) read by this output are on {}",
            against
                .as_ref()
                .map(|l| l.file.display().to_string())
                .unwrap_or_default()
        );
    }
    if failed {
        bail!("verification FAILED");
    }
    if unverifiable > 0 {
        println!("Verified: output matches; {unverifiable} input(s) could not be checked");
    } else {
        println!("Verified: output and all inputs match the record");
    }
    Ok(())
}

/// Recompute the array digest as the writer defined it: single band via
/// the file's own dtype, stacks band-sequentially in the recorded dtype.
fn output_digest(path: &Path, dtype: &str, bands: usize) -> Result<String> {
    if bands <= 1 {
        let any = read_geotiff_any(path, None).context("Failed to read raster")?;
        let digest = dispatch_any!(&any, r => raster_data_hash(&[r]).data_blake3);
        return Ok(digest);
    }
    macro_rules! stack {
        ($t:ty) => {{
            let bands: Vec<Raster<$t>> =
                read_geotiff_bands(path).context("Failed to read bands")?;
            let refs: Vec<&Raster<$t>> = bands.iter().collect();
            Ok(raster_data_hash(&refs).data_blake3)
        }};
    }
    match dtype {
        "u8" => stack!(u8),
        "u16" => stack!(u16),
        "i16" => stack!(i16),
        "u32" => stack!(u32),
        "i32" => stack!(i32),
        "f32" => stack!(f32),
        "f64" => stack!(f64),
        other => bail!("unsupported recorded dtype '{other}'"),
    }
}

/// Resolve a recorded input path: as given, else relative to the recorded
/// working directory.
fn locate(source: &str, cwd: Option<&str>) -> Option<PathBuf> {
    let p = PathBuf::from(source);
    if p.is_file() {
        return Some(p);
    }
    if !p.is_absolute()
        && let Some(cwd) = cwd
    {
        let q = Path::new(cwd).join(&p);
        if q.is_file() {
            return Some(q);
        }
    }
    None
}

#[cfg(test)]
mod affected_tests {
    use super::*;

    #[test]
    fn parse_utc_handles_dates_offsets_and_fractions() {
        assert_eq!(parse_utc("1970-01-01", false).unwrap(), 0);
        assert_eq!(parse_utc("1970-01-01", true).unwrap(), 86_399);
        assert_eq!(
            parse_utc("2026-10-26T12:07:00Z", false).unwrap(),
            1_793_016_420
        );
        assert_eq!(
            parse_utc("2026-10-26T12:07:00.000Z", false).unwrap(),
            parse_utc("2026-10-26T09:07:00-03:00", false).unwrap()
        );
        assert!(parse_utc("26/10/2026", false).is_err());
    }

    #[test]
    fn affected_ids_accept_urls_and_comments() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("l.txt");
        std::fs::write(
            &f,
            "# PC issue 516\nA\n\nhttps://pc/api/stac/v1/collections/sentinel-2-l2a/items/B?x=1\nsentinel-2-l2a/C\n",
        )
        .unwrap();
        let ids = read_affected_ids(&f).unwrap();
        assert_eq!(ids.len(), 3);
        assert!(ids.contains("A") && ids.contains("B") && ids.contains("C"));
    }
}
