﻿use crate::{
    expand_indices, idim, idx_strides,
    operator::{Broadcast, Slice, SliceDim, Split, Transpose},
    pattern::Pattern,
    udim, DataType, Operator, Shape,
};
use nalgebra::{DVector, DVectorView};
use rayon::iter::*;
use smallvec::SmallVec;
use std::{
    iter::zip,
    ops::{Deref, DerefMut},
};

#[derive(Clone, Debug)]
pub struct Tensor<Physical> {
    data_type: DataType,
    shape: Shape,
    pattern: Pattern,
    physical: Physical,
}

impl<Physical> Tensor<Physical> {
    pub fn new(data_type: DataType, shape: &[udim], physical: Physical) -> Self {
        let shape = Shape::from_iter(shape.iter().map(|&d| d as udim));
        Self {
            data_type,
            pattern: Pattern::from_shape(&shape, 0),
            shape,
            physical,
        }
    }

    /// # Safety
    ///
    /// The caller must ensure that the parts are valid.
    #[inline]
    pub unsafe fn from_raw_parts(
        data_type: DataType,
        shape: &[udim],
        pattern: &[idim],
        physical: Physical,
    ) -> Self {
        Self {
            data_type,
            shape: shape.iter().copied().collect(),
            pattern: Pattern(DVector::from_vec(pattern.to_vec())),
            physical,
        }
    }

    #[inline]
    pub const fn data_type(&self) -> DataType {
        self.data_type
    }

    #[inline]
    pub fn shape(&self) -> &[udim] {
        &self.shape
    }

    #[inline]
    pub fn pattern(&self) -> &[idim] {
        self.pattern.0.as_slice()
    }

    #[inline]
    pub fn strides(&self) -> &[idim] {
        self.pattern.strides()
    }

    #[inline]
    pub fn bytes_offset(&self) -> isize {
        self.pattern.offset() as isize * self.data_type.size() as isize
    }

    #[inline]
    pub const fn physical(&self) -> &Physical {
        &self.physical
    }

    #[inline]
    pub fn physical_mut(&mut self) -> &mut Physical {
        &mut self.physical
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.shape.iter().map(|&d| d as usize).product()
    }

    #[inline]
    pub fn bytes_size(&self) -> usize {
        self.size() * self.data_type.size()
    }

    #[inline]
    pub fn is_contiguous(&self) -> bool {
        self.contiguous_len() == self.shape.len()
    }

    /// 连续维度的数量。
    pub fn contiguous_len(&self) -> usize {
        self.pattern
            .strides()
            .iter()
            .enumerate()
            .rev()
            .scan(1 as idim, |mul, (i, &s)| {
                if s == *mul || s == 0 {
                    *mul *= self.shape[i] as idim;
                    Some(())
                } else {
                    None
                }
            })
            .count()
    }

    /// # Safety
    ///
    /// The caller must ensure that the new `physical` matches data_type, shape and pattern of `self`.
    #[inline]
    pub unsafe fn map_physical<U>(&self, f: impl FnOnce(&Physical) -> U) -> Tensor<U> {
        Tensor {
            data_type: self.data_type,
            shape: self.shape.clone(),
            pattern: self.pattern.clone(),
            physical: f(&self.physical),
        }
    }

    #[inline]
    fn byte_offset(&self) -> usize {
        self.pattern.offset() as usize * self.data_type.size()
    }

    pub fn reshape(self, shape: &[udim]) -> Self {
        assert_eq!(self.size() as udim, shape.iter().product::<udim>());
        if self.is_contiguous() {
            // reshape: 张量物理连续，直接修改形状和模式
            assert_eq!(
                self.shape.iter().product::<udim>(),
                shape.iter().product::<udim>(),
            );
            return Self {
                data_type: self.data_type,
                shape: Shape::from_slice(shape),
                pattern: Pattern::from_shape(shape, self.pattern.offset()),
                physical: self.physical,
            };
        }

        fn remove1(shape: &[udim]) -> Shape {
            shape.iter().filter(|&&d| d > 1).copied().collect::<Shape>()
        }

        let current = remove1(&self.shape);
        let target = remove1(shape);
        let same_head = zip(&current, &target).take_while(|(a, b)| a == b).count();
        if same_head == current.len() {
            // squeeze: 张量形状只增减一些 1，扫描，将新增的 1 模式设置 0
            let mut i = 0;
            let mut pattern = Vec::with_capacity(shape.len() + 1);
            for &d in shape {
                if d == 1 {
                    pattern.push(0);
                } else {
                    pattern.push(loop {
                        match self.shape[i] {
                            1 => i += 1,
                            _ => break self.pattern.0[i],
                        }
                    });
                    debug_assert_eq!(self.shape[i], d);
                    i += 1;
                }
            }
            pattern.push(self.pattern.offset());
            return Self {
                data_type: self.data_type,
                shape: Shape::from_slice(shape),
                pattern: Pattern(DVector::from_vec(pattern)),
                physical: self.physical,
            };
        }

        let same_tail = zip(
            current[same_head..].iter().rev(),
            target[same_head..].iter().rev(),
        )
        .take_while(|(a, b)| a == b)
        .count();
        if same_head + same_tail + 1 == current.len() {
            // split: 原本的一个维度拆成多个，支持拆分物理连续的那一个维度
            let axis = same_head;
            let insert_dims = &target[axis..target.len() - same_tail];

            let mut i = 0;
            let mut j = 0;
            let mut k = 0;
            let mut pattern = Vec::with_capacity(shape.len() + 1);
            while j < same_head {
                let d = shape[k];
                k += 1;
                if d == 1 {
                    pattern.push(0);
                } else {
                    pattern.push(loop {
                        match self.shape[i] {
                            1 => i += 1,
                            _ => break self.pattern.0[i],
                        }
                    });
                    debug_assert_eq!(self.shape[i], d);
                    debug_assert_eq!(current[j], d);
                    debug_assert_eq!(target[j], d);
                    i += 1;
                    j += 1;
                }
            }

            while self.shape[i] == 1 {
                i += 1;
            }
            assert_eq!(self.pattern.0[i], 1);
            i += 1;

            let (_, insert_pattern) = idx_strides(insert_dims);
            let mut l = 0;
            while j < same_head + insert_dims.len() {
                let d = shape[k];
                k += 1;
                if d == 1 {
                    pattern.push(0);
                } else {
                    pattern.push(loop {
                        match insert_dims[l] {
                            1 => l += 1,
                            _ => break insert_pattern[l] as idim,
                        }
                    });
                    debug_assert_eq!(insert_dims[l], d);
                    debug_assert_eq!(target[j], d);
                    l += 1;
                    j += 1;
                }
            }

            while k < shape.len() {
                let d = shape[k];
                k += 1;
                if d == 1 {
                    pattern.push(0);
                } else {
                    pattern.push(loop {
                        match self.shape[i] {
                            1 => i += 1,
                            _ => break self.pattern.0[i],
                        }
                    });
                    debug_assert_eq!(self.shape[i], d);
                    debug_assert_eq!(current[j], d);
                    debug_assert_eq!(target[j], d);
                    i += 1;
                    j += 1;
                }
            }

            pattern.push(self.pattern.offset());
            return Self {
                data_type: self.data_type,
                shape: Shape::from_slice(shape),
                pattern: Pattern(DVector::from_vec(pattern)),
                physical: self.physical,
            };
        }
        panic!("unsupported reshape");
    }
}

impl<Physical: Clone> Tensor<Physical> {
    pub fn apply(&self, operator: &impl Operator) -> SmallVec<[Self; 1]> {
        operator
            .build(&self.shape)
            .into_iter()
            .map(|(shape, affine)| Self {
                data_type: self.data_type,
                shape,
                pattern: Pattern(affine * &self.pattern.0),
                physical: self.physical.clone(),
            })
            .collect()
    }

    pub fn broadcast(&self, shape: &[usize]) -> Self {
        self.apply(&Broadcast(Shape::from_iter(
            shape.iter().map(|&d| d as udim),
        )))
        .into_iter()
        .next()
        .unwrap()
    }

    pub fn slice(&self, dims: &[SliceDim]) -> Self {
        self.apply(&Slice(dims.to_vec()))
            .into_iter()
            .next()
            .unwrap()
    }

    pub fn split(&self, axis: usize, segments: &[usize]) -> SmallVec<[Self; 1]> {
        self.apply(&Split {
            axis: axis as udim,
            segments: Shape::from_iter(segments.iter().map(|&d| d as udim)),
        })
    }

    pub fn transpose(&self, axes: &[usize]) -> Self {
        self.apply(&Transpose(SmallVec::from_iter(
            axes.iter().map(|&i| i as udim),
        )))
        .into_iter()
        .next()
        .unwrap()
    }
}

pub trait Storage {
    type Raw: ?Sized;
    type Access<'a>: Deref<Target = Self::Raw>
    where
        Self: 'a;
    type AccessMut<'a>: DerefMut<Target = Self::Raw>
    where
        Self: 'a;

    unsafe fn get_unchecked(&self) -> &Self::Raw;
    unsafe fn get_unchecked_mut(&mut self) -> &mut Self::Raw;
    fn access(&self) -> Self::Access<'_>;
    fn access_mut(&mut self) -> Self::AccessMut<'_>;
}

impl<Physical: Storage> Tensor<Physical> {
    #[inline]
    pub unsafe fn access_unchecked(&self) -> Tensor<&Physical::Raw> {
        Tensor {
            data_type: self.data_type,
            shape: self.shape.clone(),
            pattern: self.pattern.clone(),
            physical: self.physical.get_unchecked(),
        }
    }

    #[inline]
    pub unsafe fn access_unchecked_mut(&mut self) -> Tensor<&mut Physical::Raw> {
        Tensor {
            data_type: self.data_type,
            shape: self.shape.clone(),
            pattern: self.pattern.clone(),
            physical: self.physical.get_unchecked_mut(),
        }
    }

    #[inline]
    pub fn access(&self) -> Tensor<Physical::Access<'_>> {
        Tensor {
            data_type: self.data_type,
            shape: self.shape.clone(),
            pattern: self.pattern.clone(),
            physical: self.physical.access(),
        }
    }

    #[inline]
    pub fn access_mut(&mut self) -> Tensor<Physical::AccessMut<'_>> {
        Tensor {
            data_type: self.data_type,
            shape: self.shape.clone(),
            pattern: self.pattern.clone(),
            physical: self.physical.access_mut(),
        }
    }
}

impl<Physical: Deref<Target = [u8]>> Tensor<Physical> {
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        debug_assert!(self.is_contiguous());
        let off = self.byte_offset();
        let len = self.bytes_size();
        &self.physical[off..][..len]
    }

    pub fn locate_start(&self) -> *const u8 {
        let off = self.byte_offset();
        (&self.physical[off]) as _
    }

    pub fn locate(&self, indices: &DVectorView<idim>) -> Option<*const u8> {
        let i = self.pattern.0.dot(indices) as usize * self.data_type.size();
        self.physical.get(i).map(|r| r as _)
    }

    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        let ptr = self.physical.as_ptr();
        let offset = self.byte_offset();
        unsafe { ptr.add(offset) }
    }

    /// # Safety
    ///
    /// The caller must ensure that the `dst` can be a valid tensor physical.
    pub unsafe fn reform_to_raw(&self, dst: &mut [u8]) {
        let src = &self.physical[self.byte_offset()..];
        // 计算结尾连续维度数量
        let contiguous = self.contiguous_len();
        if contiguous == self.shape.len() {
            // 所有维度都连续，直接拷贝所有数据
            dst.copy_from_slice(&src[..dst.len()]);
        } else {
            let dt = self.data_type.size();
            // 一部分维度连续，迭代不连续的部分
            let (iter, contiguous) = self.shape.split_at(self.shape.len() - contiguous);
            let (n, idx_strides) = idx_strides(iter);
            let len = contiguous.iter().product::<udim>() as usize * dt;
            let pattern = self.pattern.0.view_range(..iter.len(), ..);
            let ptr = dst.as_mut_ptr() as usize;
            (0..n).into_par_iter().for_each(|i| {
                let j = pattern.dot(&expand_indices(i, &idx_strides, &[]));
                unsafe { std::slice::from_raw_parts_mut((ptr + i as usize * len) as *mut u8, len) }
                    .copy_from_slice(&src[j as usize * dt..][..len]);
            });
        }
    }

    pub fn reform_to<U>(&self, dst: &mut Tensor<U>)
    where
        U: DerefMut<Target = [u8]>,
    {
        assert_eq!(self.data_type, dst.data_type);
        assert_eq!(self.shape, dst.shape);
        let contiguous = self.contiguous_len().min(dst.contiguous_len());
        if contiguous == self.shape.len() {
            dst.as_mut_slice().copy_from_slice(self.as_slice());
        } else {
            let dt = self.data_type.size();
            // 一部分维度连续，迭代不连续的部分
            let (iter, contiguous) = self.shape.split_at(self.shape.len() - contiguous);
            let (n, idx_strides) = idx_strides(iter);
            let src_pattern = self.pattern.0.view_range(..iter.len(), ..);
            let dst_pattern = dst.pattern.0.view_range(..iter.len(), ..);
            let src = self.locate_start() as usize;
            let dst = dst.locate_start() as usize;
            let count = contiguous.iter().product::<udim>() as usize * dt;
            (0..n).into_par_iter().for_each(|i| {
                let indices = expand_indices(i, &idx_strides, &[]);
                let src = (src + src_pattern.dot(&indices) as usize * dt) as *const u8;
                let dst = (dst + dst_pattern.dot(&indices) as usize * dt) as *mut u8;
                unsafe { std::ptr::copy_nonoverlapping(src, dst, count) };
            });
        }
    }
}

impl<Physical: DerefMut<Target = [u8]>> Tensor<Physical> {
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        debug_assert!(self.is_contiguous());
        let off = self.byte_offset();
        let len = self.bytes_size();
        &mut self.physical[off..][..len]
    }

    pub fn locate_start_mut(&mut self) -> *mut u8 {
        let off = self.byte_offset();
        (&mut self.physical[off]) as _
    }

    pub fn locate_mut(&mut self, indices: &DVectorView<idim>) -> Option<*mut u8> {
        let i = self.pattern.0.dot(indices) as usize * self.data_type.size();
        self.physical.get_mut(i).map(|r| r as _)
    }
}

#[test]
fn test() {
    let t = Tensor::new(DataType::F32, &[2, 3, 4, 5], ());
    assert_eq!(t.shape(), &[2, 3, 4, 5]);
    assert_eq!(t.pattern.0.as_slice(), &[60, 20, 5, 1, 0]);
    assert_eq!(t.contiguous_len(), 4);
    assert_eq!(t.is_contiguous(), true);

    let t = t.reshape(&[2, 3, 20]);
    assert_eq!(t.shape(), &[2, 3, 20]);
    assert_eq!(t.pattern.0.as_slice(), &[60, 20, 1, 0]);
    assert_eq!(t.contiguous_len(), 3);
    assert_eq!(t.is_contiguous(), true);

    let t = t.transpose(&[1, 0, 2]);
    assert_eq!(t.shape(), &[3, 2, 20]);
    assert_eq!(t.pattern.0.as_slice(), &[20, 60, 1, 0]);
    assert_eq!(t.contiguous_len(), 1);
    assert_eq!(t.is_contiguous(), false);

    let t = t.reshape(&[3, 1, 1, 2, 5, 1, 4, 1, 1, 1]);
    assert_eq!(t.shape(), &[3, 1, 1, 2, 5, 1, 4, 1, 1, 1]);
    assert_eq!(t.pattern.0.as_slice(), &[20, 0, 0, 60, 4, 0, 1, 0, 0, 0, 0]);
    assert_eq!(t.contiguous_len(), 6);
    assert_eq!(t.is_contiguous(), false);

    let t = t.reshape(&[3, 2, 1, 5, 2, 2]);
    assert_eq!(t.shape(), &[3, 2, 1, 5, 2, 2]);
    assert_eq!(t.pattern.0.as_slice(), &[20, 60, 0, 4, 2, 1, 0]);
    assert_eq!(t.contiguous_len(), 4);
    assert_eq!(t.is_contiguous(), false);
}
