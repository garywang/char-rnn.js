var worker = this;

function assert(condition) {
  if (!condition) {
    throw new Error("assertion failed");
  }
}

function Memory(buffer, next) {
  this.buffer = buffer;
  this.next = next;
}

Memory.prototype.malloc = function(size) {
  var arr = new Float32Array(this.buffer, this.next, size);
  this.next += size * 4;
  assert(this.next < this.buffer.byteLength);
  return arr;
}

Memory.prototype.free = function free(array) {
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

  function vecAddElems(one, two, out) {
    assert(one.length == two.length);
    assert(one.length == out.length);
    for (var n = 0; n < one.length; n++) {
      out[n] = one[n] + two[n];
    }
    return out;
  }

  function vecMultElems(one, two, out) {
    assert(one.length == two.length);
    assert(one.length == out.length);
    for (var n = 0; n < one.length; n++) {
      out[n] = one[n] * two[n];
    }
    return out;
  }

  function makeAffineTransformation(linear, shift) {
    function affine(inVec, outVec) {
      matMult(linear, inVec, outVec);
      vecAddElems(outVec, shift, outVec);
      return outVec;
    }
    affine.inLength = linear.nCols;
    affine.outLength = linear.nRows;
    return affine;
  }

  return {
    matMult: matMult,
    vecAddElems: vecAddElems,
    vecMultElems: vecMultElems,
    makeAffineTransformation: makeAffineTransformation,
  };
}

function stringToBytes(str) {
  return unescape(encodeURIComponent(str));
}

function bytesToString(bytes) {
  return decodeURIComponent(escape(bytes));
}

function LSTM(memory, linalg, params) {
  const malloc = memory.malloc.bind(memory);
  const module = {};

  module.makeState = function makeState() {
    return malloc(params.nNodes * params.nLayers * 2);
  }

  module.makeProbs = function makeProbs() {
    return malloc(params.affines[params.affines.length - 1].outLength);
  }

  module.forward = function forward(inState, byte, outState, outLogProbs) {
    // TODO: implement this
    console.log(forwardLayer(indexState(inState, 0),
                             indexState(inState, 1),
                             byteToVector(byte),
                             params.affines[0],
                             params.affines[1],
                             indexState(outState, 0),
                             indexState(outState, 1)));
  }

  function forwardLayer(prevC, prevH, x, i2h, h2h, nextC, nextH) {
    // TODO: implement this
    var all_input_sums = linalg.vecAddElems(i2h(x, malloc(4 * params.nNodes)),
                                            h2h(prevH, malloc(4 * params.nNodes)),
                                            malloc(4 * params.nNodes));
    return all_input_sums;
  }

  function indexState(state, n) {
    return state.subarray(n * params.nNodes, (n + 1) * params.nNodes);
  }

  function byteToIndex(byte) {
    return params.vocab[byte];
  }

  function indexToByte(index) {
    return params.ivocab[index];
  }

  function byteToVector(byte) {
    var vec = malloc(params.affines[0].inLength);
    vec[byteToIndex(byte)] = 1.;
    return vec;
  }

  return {forward: forward, makeState: makeState, makeProbs: makeProbs};
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

  get(path + ".dat?f", "arraybuffer", function(buffer) {
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
  params.nLayers = affines.length - 1;

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
  lstm.forward(lstm.makeState(), "a", lstm.makeState(), lstm.makeProbs());
}

//load("data");

//window.addEventListener("load", function() {
/*
  document.body.innerHTML = "load";

  var buffer = new ArrayBuffer(0x1000000);
  var next = 0;
  function malloc(size) {
    var arr = new Float32Array(buffer, next, size);
    next += size * 4;
    assert(next < 0x1000000);
    return arr;
  }

  var matrix = malloc(N * N);
  var vector = malloc(N);

  var sum = 0;
  for (var i = 0; i < N * N; i++) {
    matrix[i] = Math.random() / 250;
    sum += matrix[i];
  }
  for (var i = 0; i < N; i++) {
    vector[i] = Math.random() / N;
    sum += vector[i];
  }

  document.body.innerHTML += sum + ' ';
  start = new Date();

  var out;
  for (var iter = 0; iter < 100; iter++) {
    out = malloc(N);
    matMult(matrix, vector, out);
    //vector = out;
  }
  end = new Date();
  document.body.innerHTML += end - start;
  document.body.innerHTML += " ";
  document.body.innerHTML += out[Math.floor(Math.random() * N)];

  var mod = AsmModule(window, null, buffer);
  start = new Date();
  var out2 = malloc(N);
  console.log(mod.matMult(matrix.byteOffset, vector.byteOffset, out2.byteOffset, N, N));
  for (var iter = 0; iter < 100; iter++) {
    mod.matMult(matrix.byteOffset, vector.byteOffset, out2.byteOffset, N, N);
  }
  end = new Date();
  console.log(end - start);
  console.log(out);
  console.log(out2);
  window.matrix = matrix;
  window.vector = vector;
  window.out2 = out2;
*/
//});
