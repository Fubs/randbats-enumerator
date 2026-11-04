//! The eight-lane vectors slot 6 computes with, on stable Rust: the subset of
//! `std::simd` it uses, lane by lane, with the same results. `mul_add` is
//! fused and `reduce_sum` adds the lanes in order, as `std::simd` does.
use std::marker::PhantomData;
use std::ops::{
    Add, AddAssign, BitAnd, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign,
};
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Debug, PartialEq, Default)]
#[repr(align(64))]
pub struct f64x8([f64; 8]);
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
#[repr(align(64))]
pub struct u64x8([u64; 8]);
/// A lane mask; `T` names the lane width, as in `std::simd`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Mask<T, const N: usize>([bool; N], PhantomData<T>);
impl f64x8 {
    #[inline(always)]
    pub const fn splat(x: f64) -> Self {
        Self([x; 8])
    }
    #[inline(always)]
    pub const fn from_array(a: [f64; 8]) -> Self {
        Self(a)
    }
    #[inline(always)]
    pub const fn to_array(self) -> [f64; 8] {
        self.0
    }
    #[inline(always)]
    pub fn from_slice(s: &[f64]) -> Self {
        Self(s[..8].try_into().unwrap())
    }
    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [f64]) {
        s[..8].copy_from_slice(&self.0);
    }
    #[inline(always)]
    fn map(self, f: impl Fn(f64) -> f64) -> Self {
        Self(self.0.map(f))
    }
    #[inline(always)]
    fn zip(self, o: Self, f: impl Fn(f64, f64) -> f64) -> Self {
        Self(std::array::from_fn(|i| f(self.0[i], o.0[i])))
    }
}
impl u64x8 {
    #[inline(always)]
    pub const fn splat(x: u64) -> Self {
        Self([x; 8])
    }
    #[inline(always)]
    pub const fn from_array(a: [u64; 8]) -> Self {
        Self(a)
    }
}
impl<T, const N: usize> Mask<T, N> {
    #[inline(always)]
    pub const fn splat(b: bool) -> Self {
        Self([b; N], PhantomData)
    }
    #[inline(always)]
    pub const fn from_array(a: [bool; N]) -> Self {
        Self(a, PhantomData)
    }
    #[inline(always)]
    pub fn cast<U>(self) -> Mask<U, N> {
        Mask(self.0, PhantomData)
    }
}
impl<T, const N: usize> BitAnd for Mask<T, N> {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, o: Self) -> Self {
        Self(std::array::from_fn(|i| self.0[i] & o.0[i]), PhantomData)
    }
}
impl BitAnd for u64x8 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, o: Self) -> Self {
        Self(std::array::from_fn(|i| self.0[i] & o.0[i]))
    }
}
macro_rules! binop {
    ($tr:ident, $f:ident, $atr:ident, $af:ident, $op:tt) => {
        impl $tr for f64x8 { type Output = Self; #[inline(always)] fn $f (self, o : Self)
        -> Self { self.zip(o, | a, b | a $op b) } } impl $atr for f64x8 {
        #[inline(always)] fn $af (& mut self, o : Self) { * self = * self $op o; } }
    };
}
binop!(Add, add, AddAssign, add_assign, +);
binop!(Sub, sub, SubAssign, sub_assign, -);
binop!(Mul, mul, MulAssign, mul_assign, *);
binop!(Div, div, DivAssign, div_assign, /);
impl Neg for f64x8 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        self.map(|a| -a)
    }
}
/// `std::simd::StdFloat`.
pub trait StdFloat {
    /// `self * a + b`, rounded once.
    fn mul_add(self, a: Self, b: Self) -> Self;
}
impl StdFloat for f64x8 {
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        Self(std::array::from_fn(|i| self.0[i].mul_add(a.0[i], b.0[i])))
    }
}
pub mod num {
    use super::f64x8;
    /// `std::simd::num::SimdFloat`.
    pub trait SimdFloat {
        fn abs(self) -> Self;
        fn reduce_sum(self) -> f64;
    }
    impl SimdFloat for f64x8 {
        #[inline(always)]
        fn abs(self) -> Self {
            self.map(f64::abs)
        }
        #[inline(always)]
        fn reduce_sum(self) -> f64 {
            self.0.iter().fold(-0.0, |s, &x| s + x)
        }
    }
}
pub mod cmp {
    use super::{Mask, f64x8, u64x8};
    /// `std::simd::cmp::SimdPartialEq`.
    pub trait SimdPartialEq {
        type Mask;
        fn simd_eq(self, o: Self) -> Self::Mask;
    }
    /// `std::simd::cmp::SimdPartialOrd`.
    pub trait SimdPartialOrd {
        type Mask;
        fn simd_ge(self, o: Self) -> Self::Mask;
    }
    impl SimdPartialEq for u64x8 {
        type Mask = Mask<i64, 8>;
        #[inline(always)]
        fn simd_eq(self, o: Self) -> Mask<i64, 8> {
            Mask::from_array(std::array::from_fn(|i| self.0[i] == o.0[i]))
        }
    }
    impl SimdPartialOrd for f64x8 {
        type Mask = Mask<i64, 8>;
        #[inline(always)]
        fn simd_ge(self, o: Self) -> Mask<i64, 8> {
            Mask::from_array(std::array::from_fn(|i| self.0[i] >= o.0[i]))
        }
    }
}
/// `std::simd::Select`.
pub trait Select<V> {
    fn select(self, t: V, f: V) -> V;
}
impl Select<f64x8> for Mask<i64, 8> {
    #[inline(always)]
    fn select(self, t: f64x8, f: f64x8) -> f64x8 {
        f64x8(std::array::from_fn(|i| if self.0[i] { t.0[i] } else { f.0[i] }))
    }
}
