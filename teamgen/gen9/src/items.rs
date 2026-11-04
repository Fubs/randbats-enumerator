//! Item data.
use crate::generated::item_id::ItemId;
#[derive(Copy, Clone, Debug)]
pub struct ItemMetadata {
    pub id: ItemId,
    pub name: &'static str,
}
