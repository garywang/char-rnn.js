var worker = this;

function assert(condition) {
  if (!condition) {
    throw new Error("assertion failed");
  }
}

function Memory(buffer, next) {
  this.buffer = buffer;
  this.next = next;
  this.stack = [];
}

Memory.prototype.malloc = function(size) {
  var arr = new Float32Array(this.buffer, this.next, size);
  this.next += size * 4;
  assert(this.next < this.buffer.byteLength);
  return arr;
}

Memory.prototype.pushFrame = function() {
  this.stack.push(this.next);
}

Memory.prototype.popFrame = function free(array) {
  this.next = this.stack.pop();
  assert(this.next !== undefined);
}

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

  var matMultAsm = AsmModule(worker, null, memory.buffer).matMult;

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

function stringToBytes(str) {
  return unescape(encodeURIComponent(str));
}

function bytesToString(bytes) {
  return decodeURIComponent(escape(bytes));
}

// See https://github.com/karpathy/char-rnn/blob/master/model/LSTM.lua
function LSTM(memory, linalg, params) {
  const malloc = memory.malloc.bind(memory);
  const module = {};

  function makeState() {
    var state = malloc(params.nNodes * params.nLayers * 2);
    linalg.zero(state, state);
    return state;
  }
  module.makeState = makeState;

  function makeProbs() {
    return malloc(params.affines[params.affines.length - 1].outLength);
  }
  module.makeProbs = makeProbs;

  function forward(inState, byte, outState, outProbs) {
    memory.pushFrame();

    var input = byteToVector(byte);
    for (var n = 0; n < params.nLayers; n++) {
      input = forwardLayer(indexState(inState, 2 * n),
                           indexState(inState, 2 * n + 1),
                           input,
                           params.affines[2 * n],
                           params.affines[2 * n + 1],
                           indexState(outState, 2 * n),
                           indexState(outState, 2 * n + 1));
    }
    var topH = indexState(outState, 2 * params.nLayers - 1);
    params.affines[2 * params.nLayers](topH, outProbs);
    linalg.exp(outProbs, outProbs);
    normalize(outProbs);

    memory.popFrame();
    return outProbs;
  }
  module.forward = forward;

  function forwardLayer(prevC, prevH, x, i2h, h2h, nextC, nextH) {
    memory.pushFrame();

    var allInputSums = linalg.vecAddElems(i2h(x), h2h(prevH));

    var inGate = linalg.sigmoid(indexState(allInputSums, 0));
    var forgetGate = linalg.sigmoid(indexState(allInputSums, 1));
    var outGate = linalg.sigmoid(indexState(allInputSums, 2));
    var inTransform = linalg.tanh(indexState(allInputSums, 3));

    nextC = linalg.vecAddElems(linalg.vecMultElems(forgetGate, prevC),
                               linalg.vecMultElems(inGate, inTransform));
    nextH = linalg.vecMultElems(outGate, linalg.tanh(nextC));

    memory.popFrame();
    return nextC;
  }

  function indexState(state, n) {
    return state.subarray(n * params.nNodes, (n + 1) * params.nNodes);
  }

  function normalize(probs) {
    var sum = probs.reduce((x, y) => x + y);
    linalg.scalarMult(probs, 1 / sum, probs);
    return probs;
  }

  function byteToIndex(byte) {
    return params.vocab[byte];
  }
  module.byteToIndex = byteToIndex;

  function indexToByte(index) {
    return params.ivocab[index];
  }
  module.byteToIndex = byteToIndex;

  function byteToVector(byte) {
    var vec = malloc(params.affines[0].inLength);
    vec[byteToIndex(byte)] = 1.;
    return vec;
  }

  return module;
}

function load(path) {
  function get(url, type, callback) {
    var req = new XMLHttpRequest();
    req.open("GET", url, true);
    req.responseType = type;
    req.onload = function() {
      callback(req.response);
    }
    req.send();
    return req;
  }

  get(path + ".dat", "arraybuffer", function(buffer) {
    get(path + ".json", "json", function(metadata) {
      console.log("Got data");
      init(buffer, metadata);
    });
  });
}

function init(buffer, metadata) {
  var memory = new Memory(buffer, metadata.next);
  var linalg = new Linalg(memory);

  var arrays = []
  for (var n = 0; n < metadata.arrays.length; n++) {
    var array =
      new Float32Array(buffer, metadata.arrays[n].offset, metadata.arrays[n].length);
    var dims = metadata.arrays[n].dim;
    if (dims.length > 1) {
      array.nRows = dims[0];
      array.nCols = dims[1];
    }
    array.dims = dims;
    arrays.push(array);
  }

  var affines = [];
  for (var n = 0; n < arrays.length / 2; n++) {
    affines.push(linalg.makeAffineTransformation(arrays[2 * n], arrays[2 * n + 1]));
  }

  var params = {};
  params.affines = affines;
  params.nNodes = affines[0].outLength / 4;
  params.nLayers = (affines.length - 1) / 2;

  params.vocab = {};
  params.ivocab = {};
  for (var n = 0; n < metadata.vocab.length; n++) {
    if (metadata.vocab[n] !== null) {
      params.vocab[String.fromCharCode(n)] = metadata.vocab[n];
      params.ivocab[metadata.vocab[n]] = String.fromCharCode(n);
    }
  }

  worker.inited = true;
  console.log("inited");

  var lstm = LSTM(memory, linalg, params);

  var d = new Date();
  for (var n = 0; n < 100; n++) {
    lstm.forward(lstm.makeState(), "a", lstm.makeState(), lstm.makeProbs());
  }
  console.log(new Date() - d);
  console.log(lstm.forward(lstm.makeState(), "a", lstm.makeState(), lstm.makeProbs()));
}

//load("data");
