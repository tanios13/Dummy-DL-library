# Auto Diff library

Implementing a simple autodiff library to get an insight of how Deep Learning libraries like TensorFlow and PyTorch works.

## Summary

- Very easy to make sure everything is correct
  - Can test individual functions independenlty
- Easy to extend to extra operations
- Flexible:
  - e.g. could implement some operations for GPU
- Limitations (still to implement):
  - Allow batches (multiple samples)
  - More operations: e.g. matmul
  - Allow multiple differentiation of multiple variable (n dimension of data)
  - Use graphs instead of requiring lists for defining models
  - Make it cleaner: hide passing grad_f functions from user


