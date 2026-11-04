//! Pokémon Showdown's Gen 9 Random Battles team generator, ported to Rust
//! draw for draw, with the data tables it reads.
//!
//! The tables are generated at build time from a dump of the vendored
//! Pokémon Showdown (`node dump.js`, see `build.rs`); [`PS_COMMIT`],
//! [`DATA_FINGERPRINT`] and [`LOGIC_FINGERPRINT`] say which.
pub mod abilities;
pub mod items;
pub mod moves;
pub mod pokemon;
pub mod randbats;
pub mod randbats_gen;
pub mod species;
pub mod types;
/// Tables generated from the dump.
pub mod generated {
    pub mod ability_id {
        include!(concat!(env!("OUT_DIR"), "/ability_id.rs"));
    }
    pub mod item_id {
        include!(concat!(env!("OUT_DIR"), "/item_id.rs"));
    }
    pub mod move_id {
        include!(concat!(env!("OUT_DIR"), "/move_id.rs"));
    }
    pub mod species_id {
        include!(concat!(env!("OUT_DIR"), "/species_id.rs"));
    }
    pub mod abilities_data {
        include!(concat!(env!("OUT_DIR"), "/abilities_data.rs"));
    }
    pub mod items_data {
        include!(concat!(env!("OUT_DIR"), "/items_data.rs"));
    }
    pub mod moves_data {
        include!(concat!(env!("OUT_DIR"), "/moves_data.rs"));
    }
    pub mod species_data {
        include!(concat!(env!("OUT_DIR"), "/species_data.rs"));
    }
    pub mod randbats_data {
        include!(concat!(env!("OUT_DIR"), "/randbats_data.rs"));
    }
    pub mod type_chart {
        include!(concat!(env!("OUT_DIR"), "/type_chart.rs"));
    }
    pub(crate) mod source {
        include!(concat!(env!("OUT_DIR"), "/source.rs"));
    }
}
/// The Pokémon Showdown commit the tables were dumped from.
pub const PS_COMMIT: &str = generated::source::PS_COMMIT;
/// SHA-256 of the data the generator reads (`dump.js`'s `fingerprint`).
pub const DATA_FINGERPRINT: &str = generated::source::DATA_FINGERPRINT;
/// SHA-256 of the PS sources the port translates (`dump.js`'s `logic`).
pub const LOGIC_FINGERPRINT: &str = generated::source::LOGIC_FINGERPRINT;
/// PS's `toID`: lowercase, alphanumeric only.
pub fn to_id(text: &str) -> String {
    text.chars()
        .flat_map(char::to_lowercase)
        .filter(|c| c.is_ascii_alphanumeric())
        .collect()
}
