//! Minimal zero-dependency JSON parser and serializer.

use std::collections::BTreeMap;
use std::fmt::Write as _;

#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(f64, String),
    String(String),
    Array(Vec<JsonValue>),
    Object(BTreeMap<String, JsonValue>),
}

impl JsonValue {
    pub fn obj(entries: Vec<(&str, JsonValue)>) -> Self {
        let mut map = BTreeMap::new();
        for (k, v) in entries {
            map.insert(k.to_string(), v);
        }
        JsonValue::Object(map)
    }

    pub fn num_f64(v: f64) -> Self {
        if v.fract() == 0.0 && v.abs() < 1e15 {
            JsonValue::Number(v, format!("{:.1}", v))
        } else {
            JsonValue::Number(v, format!("{}", v))
        }
    }

    pub fn num_u64(v: u64) -> Self {
        JsonValue::Number(v as f64, v.to_string())
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            JsonValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            JsonValue::Number(n, _) => Some(*n),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            JsonValue::Number(n, raw) => raw.parse::<u64>().ok().or_else(|| {
                if *n >= 0.0 && n.fract() == 0.0 {
                    Some(*n as u64)
                } else {
                    None
                }
            }),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&Vec<JsonValue>> {
        match self {
            JsonValue::Array(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&BTreeMap<String, JsonValue>> {
        match self {
            JsonValue::Object(m) => Some(m),
            _ => None,
        }
    }

    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        self.as_object().and_then(|m| m.get(key))
    }

    pub fn is_null(&self) -> bool {
        matches!(self, JsonValue::Null)
    }

    pub fn to_json_string(&self, pretty: bool) -> String {
        let mut out = String::new();
        self.write_json(&mut out, pretty, 0);
        out
    }

    fn write_json(&self, out: &mut String, pretty: bool, indent: usize) {
        match self {
            JsonValue::Null => out.push_str("null"),
            JsonValue::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            JsonValue::Number(_, raw) => out.push_str(raw),
            JsonValue::String(s) => write_escaped_str(out, s),
            JsonValue::Array(items) => {
                if items.is_empty() {
                    out.push_str("[]");
                    return;
                }
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    if pretty {
                        out.push('\n');
                        push_indent(out, indent + 2);
                    }
                    item.write_json(out, pretty, indent + 2);
                }
                if pretty {
                    out.push('\n');
                    push_indent(out, indent);
                }
                out.push(']');
            }
            JsonValue::Object(map) => {
                if map.is_empty() {
                    out.push_str("{}");
                    return;
                }
                out.push('{');
                for (i, (k, v)) in map.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    if pretty {
                        out.push('\n');
                        push_indent(out, indent + 2);
                    }
                    write_escaped_str(out, k);
                    out.push_str(if pretty { ": " } else { ":" });
                    v.write_json(out, pretty, indent + 2);
                }
                if pretty {
                    out.push('\n');
                    push_indent(out, indent);
                }
                out.push('}');
            }
        }
    }
}

fn push_indent(out: &mut String, n: usize) {
    for _ in 0..n {
        out.push(' ');
    }
}

fn write_escaped_str(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\x08' => out.push_str("\\b"),
            '\x0c' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

struct Parser<'a> {
    src: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn skip_ws(&mut self) {
        while self.pos < self.src.len() && matches!(self.src[self.pos], b' ' | b'\t' | b'\n' | b'\r') {
            self.pos += 1;
        }
    }

    fn parse_value(&mut self) -> Result<JsonValue, String> {
        self.skip_ws();
        if self.pos >= self.src.len() {
            return Err("unexpected end of JSON".to_string());
        }
        match self.src[self.pos] {
            b'n' => self.expect_bytes(b"null").map(|_| JsonValue::Null),
            b't' => self.expect_bytes(b"true").map(|_| JsonValue::Bool(true)),
            b'f' => self.expect_bytes(b"false").map(|_| JsonValue::Bool(false)),
            b'"' => self.parse_string().map(JsonValue::String),
            b'[' => self.parse_array(),
            b'{' => self.parse_object(),
            b'-' | b'0'..=b'9' => self.parse_number(),
            c => Err(format!("unexpected byte '{}' at offset {}", c as char, self.pos)),
        }
    }

    fn expect_bytes(&mut self, lit: &[u8]) -> Result<(), String> {
        if self.src[self.pos..].starts_with(lit) {
            self.pos += lit.len();
            Ok(())
        } else {
            Err(format!("invalid JSON literal at offset {}", self.pos))
        }
    }

    fn parse_string(&mut self) -> Result<String, String> {
        self.pos += 1; // opening quote
        let mut out = String::new();
        while self.pos < self.src.len() {
            let b = self.src[self.pos];
            if b == b'"' {
                self.pos += 1;
                return Ok(out);
            }
            if b == b'\\' {
                self.pos += 1;
                if self.pos >= self.src.len() {
                    return Err("unterminated escape sequence".to_string());
                }
                let esc = self.src[self.pos];
                self.pos += 1;
                match esc {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\x08'),
                    b'f' => out.push('\x0c'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        let code = self.parse_hex4()?;
                        if (0xD800..=0xDBFF).contains(&code)
                            && self.pos + 2 <= self.src.len()
                            && &self.src[self.pos..self.pos + 2] == b"\\u"
                        {
                            self.pos += 2;
                            let low = self.parse_hex4()?;
                            let cp = 0x10000 + (((code as u32 - 0xD800) << 10) | (low as u32 - 0xDC00));
                            let ch = char::from_u32(cp).ok_or_else(|| "invalid surrogate pair".to_string())?;
                            out.push(ch);
                        } else {
                            let ch = char::from_u32(code as u32).ok_or_else(|| "invalid \\u escape".to_string())?;
                            out.push(ch);
                        }
                    }
                    other => return Err(format!("invalid JSON escape '\\{}'", other as char)),
                }
            } else if b < 0x80 {
                out.push(b as char);
                self.pos += 1;
            } else {
                let s = std::str::from_utf8(&self.src[self.pos..])
                    .map_err(|_| "invalid UTF-8 in JSON string".to_string())?;
                let ch = s.chars().next().ok_or_else(|| "unexpected EOF".to_string())?;
                out.push(ch);
                self.pos += ch.len_utf8();
            }
        }
        Err("unterminated JSON string".to_string())
    }

    fn parse_hex4(&mut self) -> Result<u16, String> {
        if self.pos + 4 > self.src.len() {
            return Err("truncated \\uXXXX escape".to_string());
        }
        let s = std::str::from_utf8(&self.src[self.pos..self.pos + 4])
            .map_err(|_| "invalid UTF-8 in \\u escape".to_string())?;
        self.pos += 4;
        u16::from_str_radix(s, 16).map_err(|e| format!("invalid hex in \\u escape: {e}"))
    }

    fn parse_number(&mut self) -> Result<JsonValue, String> {
        let start = self.pos;
        if self.src[self.pos] == b'-' {
            self.pos += 1;
        }
        while self.pos < self.src.len() && self.src[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        if self.pos < self.src.len() && self.src[self.pos] == b'.' {
            self.pos += 1;
            while self.pos < self.src.len() && self.src[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        }
        if self.pos < self.src.len() && matches!(self.src[self.pos], b'e' | b'E') {
            self.pos += 1;
            if self.pos < self.src.len() && matches!(self.src[self.pos], b'+' | b'-') {
                self.pos += 1;
            }
            while self.pos < self.src.len() && self.src[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        }
        let raw = std::str::from_utf8(&self.src[start..self.pos])
            .map_err(|_| "invalid number utf8".to_string())?
            .to_string();
        let val = raw.parse::<f64>().map_err(|e| format!("invalid JSON number '{raw}': {e}"))?;
        Ok(JsonValue::Number(val, raw))
    }

    fn parse_array(&mut self) -> Result<JsonValue, String> {
        self.pos += 1;
        let mut items = Vec::new();
        self.skip_ws();
        if self.pos < self.src.len() && self.src[self.pos] == b']' {
            self.pos += 1;
            return Ok(JsonValue::Array(items));
        }
        loop {
            items.push(self.parse_value()?);
            self.skip_ws();
            if self.pos >= self.src.len() {
                return Err("unterminated array".to_string());
            }
            match self.src[self.pos] {
                b',' => self.pos += 1,
                b']' => {
                    self.pos += 1;
                    return Ok(JsonValue::Array(items));
                }
                c => return Err(format!("expected ',' or ']' in array, got '{}'", c as char)),
            }
        }
    }

    fn parse_object(&mut self) -> Result<JsonValue, String> {
        self.pos += 1;
        let mut map = BTreeMap::new();
        self.skip_ws();
        if self.pos < self.src.len() && self.src[self.pos] == b'}' {
            self.pos += 1;
            return Ok(JsonValue::Object(map));
        }
        loop {
            self.skip_ws();
            if self.pos >= self.src.len() || self.src[self.pos] != b'"' {
                return Err(format!("expected string key at offset {}", self.pos));
            }
            let key = self.parse_string()?;
            self.skip_ws();
            if self.pos >= self.src.len() || self.src[self.pos] != b':' {
                return Err(format!("expected ':' after key at offset {}", self.pos));
            }
            self.pos += 1;
            let val = self.parse_value()?;
            map.insert(key, val);
            self.skip_ws();
            if self.pos >= self.src.len() {
                return Err("unterminated object".to_string());
            }
            match self.src[self.pos] {
                b',' => self.pos += 1,
                b'}' => {
                    self.pos += 1;
                    return Ok(JsonValue::Object(map));
                }
                c => return Err(format!("expected ',' or '}}' in object, got '{}'", c as char)),
            }
        }
    }
}

pub fn parse_json(text: &str) -> Result<JsonValue, String> {
    let mut p = Parser { src: text.as_bytes(), pos: 0 };
    let v = p.parse_value()?;
    p.skip_ws();
    if p.pos != p.src.len() {
        return Err(format!("trailing JSON bytes at offset {}", p.pos));
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::parse_json;

    #[test]
    fn roundtrip_complex_json() {
        let s = r#"{"a":1.25,"b":["x\n\"y\"",true,null],"nested":{"k":[1,2,3]}}"#;
        let v = parse_json(s).unwrap();
        assert_eq!(v.get("a").unwrap().as_f64().unwrap(), 1.25);
        let back = v.to_json_string(false);
        assert_eq!(parse_json(&back).unwrap(), v);
    }
}
