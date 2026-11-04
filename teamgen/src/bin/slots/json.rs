//! Objects that keep their key order, printed as `serde_json`'s pretty
//! printer prints (`serde_json` is built without `preserve_order`), and the
//! name order pkmn's `update` sorts by.
use std::cmp::Ordering;
use serde_json::Value;
pub enum Json {
    Leaf(Value),
    Object(Vec<(String, Json)>),
}
impl Json {
    pub fn pretty(&self) -> String {
        let mut out = String::new();
        self.write(&mut out, 0);
        out.push('\n');
        out
    }
    fn write(&self, out: &mut String, depth: usize) {
        match self {
            Json::Object(entries) => {
                object(out, depth, entries.iter().map(|(k, v)| (k.as_str(), Item::J(v))))
            }
            Json::Leaf(v) => leaf(out, depth, v),
        }
    }
}
enum Item<'a> {
    J(&'a Json),
    V(&'a Value),
}
fn leaf(out: &mut String, depth: usize, v: &Value) {
    match v {
        Value::Object(m) => {
            object(out, depth, m.iter().map(|(k, v)| (k.as_str(), Item::V(v))))
        }
        Value::Array(a) => {
            if a.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                out.push_str(if i == 0 { "\n" } else { ",\n" });
                indent(out, depth + 1);
                leaf(out, depth + 1, x);
            }
            out.push('\n');
            indent(out, depth);
            out.push(']');
        }
        _ => out.push_str(&v.to_string()),
    }
}
fn object<'a>(
    out: &mut String,
    depth: usize,
    entries: impl ExactSizeIterator<Item = (&'a str, Item<'a>)>,
) {
    if entries.len() == 0 {
        out.push_str("{}");
        return;
    }
    out.push('{');
    for (i, (k, v)) in entries.enumerate() {
        out.push_str(if i == 0 { "\n" } else { ",\n" });
        indent(out, depth + 1);
        out.push_str(&Value::from(k).to_string());
        out.push_str(": ");
        match v {
            Item::J(j) => j.write(out, depth + 1),
            Item::V(v) => leaf(out, depth + 1, v),
        }
    }
    out.push('\n');
    indent(out, depth);
    out.push('}');
}
fn indent(out: &mut String, depth: usize) {
    for _ in 0..depth {
        out.push_str("  ");
    }
}
/// Node's `a.localeCompare(b)` (ICU root collation, the `en-US` default) on
/// the names PS prints: case and accents only break ties, whitespace and
/// punctuation sort before digits and digits before letters. pkmn's
/// `update` orders species, and equal shares, this way; byte order differs
/// from it on case (`"a" < "B"`) and on some punctuation (`'` after `-`).
pub fn locale_cmp(a: &str, b: &str) -> Ordering {
    let (ka, kb) = (keys(a), keys(b));
    let primary = |k: &[(u32, u8, u8)]| k.iter().map(|x| x.0).collect::<Vec<_>>();
    let secondary = |k: &[(u32, u8, u8)]| k.iter().map(|x| x.1).collect::<Vec<_>>();
    let tertiary = |k: &[(u32, u8, u8)]| k.iter().map(|x| x.2).collect::<Vec<_>>();
    primary(&ka)
        .cmp(&primary(&kb))
        .then_with(|| secondary(&ka).cmp(&secondary(&kb)))
        .then_with(|| tertiary(&ka).cmp(&tertiary(&kb)))
        .then_with(|| a.cmp(b))
}
/// CLDR root's order of the ASCII punctuation and symbols.
const PUNCTUATION: &str = "_-,;:!?.'\"()[]{}@*/\\&#%`^+<=>|~$";
/// (primary, secondary, tertiary) weights per character.
fn keys(s: &str) -> Vec<(u32, u8, u8)> {
    s.chars()
        .map(|c| {
            let (base, accent) = match c {
                'á' | 'à' | 'â' | 'ä' | 'ã' => {
                    ('a', 1 + "áàâäã".chars().position(|x| x == c).unwrap() as u8)
                }
                'é' | 'è' | 'ê' | 'ë' => {
                    ('e', 1 + "éèêë".chars().position(|x| x == c).unwrap() as u8)
                }
                'É' => ('E', 1),
                'í' | 'ï' => {
                    ('i', 1 + "íï".chars().position(|x| x == c).unwrap() as u8)
                }
                'ó' | 'ö' => {
                    ('o', 1 + "óö".chars().position(|x| x == c).unwrap() as u8)
                }
                'ú' | 'ü' => {
                    ('u', 1 + "úü".chars().position(|x| x == c).unwrap() as u8)
                }
                'ñ' => ('n', 1),
                _ => (c, 0),
            };
            let upper = u8::from(base.is_ascii_uppercase());
            let primary = if base.is_whitespace() {
                0x100
            } else if let Some(i) = PUNCTUATION.find(base) {
                0x200 + i as u32
            } else if base.is_ascii_digit() {
                0x400 + base as u32
            } else if base.is_ascii_alphabetic() {
                0x500 + base.to_ascii_lowercase() as u32
            } else {
                0x10000 + base as u32
            };
            (primary, accent, upper)
        })
        .collect()
}
