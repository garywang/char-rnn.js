var assert = require("assert");

function Linalg(memory) {
  function AsmModule(stdlib, foreign, buffer) {
    "use asm";

    var imul = stdlib.Math.imul;
    var fround = stdlib.Math.fround;

    var arr = new stdlib.Float32Array(buffer);

    function matMult(mat, inVec, outVec, nRows, nCols) {
      mat = mat|0;
      inVec = inVec|0;
      outVec = outVec|0;
      nRows = nRows|0;
      nCols = nCols|0;

      var matPtr = 0, inVecPtr = 0, outVecPtr = 0;
      var matEnd = 0, inVecEnd = 0, outVecEnd = 0;

      matEnd = mat + (imul(nRows, nCols) << 2)|0;
      inVecEnd = inVec + (nCols << 2)|0;
      outVecEnd = outVec + (nRows << 2)|0;

      matPtr = mat;
      for (outVecPtr = outVec;
           (outVecPtr|0) < (outVecEnd|0);
           outVecPtr = (outVecPtr + 4)|0) {
        arr[outVecPtr >> 2] = 0.;
        for (inVecPtr = inVec;
             (inVecPtr|0) < (inVecEnd|0);
             inVecPtr = (inVecPtr + 4)|0, matPtr = (matPtr + 4)|0) {
          arr[outVecPtr >> 2] = arr[outVecPtr >> 2] +
            fround(arr[matPtr >> 2] * arr[inVecPtr >> 2]);
        }
      }
    }

    return { matMult: matMult };
  }

  var matMultAsm = AsmModule(global, null, memory.buffer).matMult;

  function matMult(matrix, inVec, outVec) {
    assert(matrix.nCols == inVec.length);
    assert(matrix.nRows == outVec.length);
    assert(matrix.buffer == memory.buffer);
    assert(inVec.buffer == memory.buffer);
    assert(outVec.buffer == memory.buffer);

    matMultAsm(matrix.byteOffset,
               inVec.byteOffset,
               outVec.byteOffset,
               matrix.nRows,
               matrix.nCols);

    return outVec;
  }

  function map1(func) {
    return function(inVec, outVec) {
      if (!outVec) {
        outVec = memory.malloc(inVec.length);
      }
      assert(inVec.length == outVec.length);
      for (var n = 0; n < inVec.length; n++) {
        outVec[n] = func(inVec[n]);
      }
      return outVec;
    }
  }

  function map2(func) {
    return function(one, two, out) {
      assert(one.length == two.length);
      if (!out) {
        out = memory.malloc(one.length);
      }
      assert(one.length == out.length);
      for (var n = 0; n < one.length; n++) {
        out[n] = func(one[n], two[n]);
      }
      return out;
    }
  }

  var add = map2((x, y) => x + y);
  var mult = map2((x, y) => x * y);

  function makeAffineTransformation(linear, shift) {
    function affine(inVec, outVec) {
      if (!outVec) {
        outVec = memory.malloc(linear.nRows);
      }
      assert(inVec.byteOffset != outVec.byteOffset);
      matMult(linear, inVec, outVec);
      add(outVec, shift, outVec);
      return outVec;
    }
    affine.inLength = linear.nCols;
    affine.outLength = linear.nRows;
    affine.linear = linear;
    affine.shift = shift;
    return affine;
  }

  function sigmoid(inVec, outVec) {
    if (!outVec) {
      outVec = memory.malloc(inVec.length);
    }
    assert(inVec.length == outVec.length);
    for (var n = 0; n < inVec.length; n++) {
      outVec[n] = 1 / (1 + Math.exp(-inVec[n]));
    }
    return outVec;
  }

  function scalarMult(inVec, scalar, outVec) {
    return (map1(x => scalar * x)(inVec, outVec));
  }

  return {
    vecAddElems: add,
    vecMultElems: mult,
    scalarMult: scalarMult,
    //sigmoid: map1(x => 1 / (1 + Math.exp(-x))),
    sigmoid: sigmoid,
    tanh: map1(Math.tanh),
    exp: map1(Math.exp),
    log: map1(Math.log),
    zero: map1(x => 0),
    makeAffineTransformation: makeAffineTransformation,
  };
}
module.exports = Linalg;
