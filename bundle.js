(function e(t,n,r){function s(o,u){if(!n[o]){if(!t[o]){var a=typeof require=="function"&&require;if(!u&&a)return a(o,!0);if(i)return i(o,!0);var f=new Error("Cannot find module '"+o+"'");throw f.code="MODULE_NOT_FOUND",f}var l=n[o]={exports:{}};t[o][0].call(l.exports,function(e){var n=t[o][1][e];return s(n?n:e)},l,l.exports,e,t,n,r)}return n[o].exports}var i=typeof require=="function"&&require;for(var o=0;o<r.length;o++)s(r[o]);return s})({1:[function(require,module,exports){
module.exports = function assert(condition) {
  if (!condition) {
    throw new Error("assertion failed");
  }
}

},{}],2:[function(require,module,exports){
(function (global){
var assert = require("./assert");

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

}).call(this,typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})

},{"./assert":1}],3:[function(require,module,exports){
// See https://github.com/karpathy/char-rnn/blob/master/model/LSTM.lua
function LSTM(memory, linalg, params) {
  const exports = {};

  function makeState() {
    var state = memory.malloc(params.nNodes * params.nLayers * 2);
    return resetState(state);
  }
  exports.makeState = makeState;

  function resetState(state) {
    linalg.zero(state, state);
    return state;
  }
  exports.resetState = resetState;

  function forward(inState, byte, outState) {
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
    memory.popFrame();
    return outState;
  }
  exports.forward = forward;

  function predict(state) {
    memory.pushFrame();
    var topH = indexState(state, 2 * params.nLayers - 1);
    var probs = params.affines[2 * params.nLayers](topH);
    probs = linalg.exp(probs, probs);
    probs = normalizeAndExport(probs);
    memory.popFrame();
    return probs;
  }
  exports.predict = predict;

  function forwardLayer(prevC, prevH, x, i2h, h2h, nextC, nextH) {
    memory.pushFrame();

    var allInputSums = linalg.vecAddElems(i2h(x), h2h(prevH));

    var inGate = linalg.sigmoid(indexState(allInputSums, 0));
    var forgetGate = linalg.sigmoid(indexState(allInputSums, 1));
    var outGate = linalg.sigmoid(indexState(allInputSums, 2));
    var inTransform = linalg.tanh(indexState(allInputSums, 3));

    linalg.vecAddElems(linalg.vecMultElems(forgetGate, prevC),
                       linalg.vecMultElems(inGate, inTransform),
                       nextC);
    linalg.vecMultElems(outGate, linalg.tanh(nextC), nextH);

    memory.popFrame();
    //console.log(nextC);
    //console.log(nextH);
    return nextH;
  }

  function indexState(state, n) {
    return state.subarray(n * params.nNodes, (n + 1) * params.nNodes);
  }

  function normalizeAndExport(probs) {
    var sum = probs.reduce((x, y) => x + y);
    var outProbs = new Float32Array(probs.length);
    return linalg.scalarMult(probs, 1 / sum, outProbs);
  }

  function byteToIndex(byte) {
    return params.vocab[byte];
  }
  exports.byteToIndex = byteToIndex;

  function indexToByte(index) {
    return params.ivocab[index];
  }
  exports.byteToIndex = byteToIndex;

  function byteToVector(byte) {
    var vec = memory.malloc(params.affines[0].inLength);
    linalg.zero(vec, vec);
    vec[byteToIndex(byte)] = 1.;
    return vec;
  }

  return exports;
}
module.exports = LSTM;

},{}],4:[function(require,module,exports){
var assert = require("./assert");

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

module.exports = function(buffer, next) {
  return new Memory(buffer, next);
}

},{"./assert":1}],5:[function(require,module,exports){
module.exports = function(memory, model) {
  "use strict"

  return function score(bytes) {
    memory.pushFrame();

    var state = model.makeState();
    var currentScore = 0;
    for (var n = 0; n < bytes.length; n++) {
      if (n > 0) {
        var probs = model.predict(state);
        currentScore += Math.log(probs[model.byteToIndex(bytes[n])]);
      }
      model.forward(state, bytes[n], state);
    }

    memory.popFrame();
    return currentScore;
  }
}

},{}],6:[function(require,module,exports){
(function (global){
var assert = require("./assert");

function stringToBytes(str) {
  return unescape(encodeURIComponent(str));
}

function bytesToString(bytes) {
  return decodeURIComponent(escape(bytes));
}

function load(path, callback) {
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

  get(path + ".dat?", "arraybuffer", function(buffer) {
    get(path + ".json", "json", function(metadata) {
      console.log("Got data");
      init(buffer, metadata);
      if (callback) {
        callback();
      }
    });
  });
}

function init(buffer, metadata) {
  var memory = require("./memory")(buffer, metadata.next);
  var linalg = require("./linalg")(memory);

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

  var model = require("./lstm")(memory, linalg, params);

  var d = new Date();
  for (var n = 0; n < 100; n++) {
    model.predict(model.forward(model.makeState(), "a", model.makeState()));
  }
  console.log(new Date() - d);
  console.log(model.predict(model.forward(model.makeState(), "a", model.makeState())));
  console.log(model.predict(model.forward(model.makeState(), "a", model.makeState())));

  global.score = require("./score")(memory, model);

  var d = new Date();
  for (var n = 0; n < 10; n++) {
    global.score(" this is some text foo bar baz");
  }
  console.log(new Date() - d);

  return { memory: memory, model: model };
}

//load("data/large");

global.load = load;

}).call(this,typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})

},{"./assert":1,"./linalg":2,"./lstm":3,"./memory":4,"./score":5}]},{},[6])
//# sourceMappingURL=data:application/json;charset:utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uLy4uLy4uL3RtcC9ub2RlLy5ucG0tcGFja2FnZXMvbGliL25vZGVfbW9kdWxlcy93YXRjaGlmeS9ub2RlX21vZHVsZXMvYnJvd3NlcmlmeS9ub2RlX21vZHVsZXMvYnJvd3Nlci1wYWNrL19wcmVsdWRlLmpzIiwiYXNzZXJ0LmpzIiwibGluYWxnLmpzIiwibHN0bS5qcyIsIm1lbW9yeS5qcyIsInNjb3JlLmpzIiwid29ya2VyLmpzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBO0FDQUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOzs7QUNMQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOzs7O0FDeElBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUMvRkE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FDM0JBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7O0FDcEJBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBIiwiZmlsZSI6ImdlbmVyYXRlZC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzQ29udGVudCI6WyIoZnVuY3Rpb24gZSh0LG4scil7ZnVuY3Rpb24gcyhvLHUpe2lmKCFuW29dKXtpZighdFtvXSl7dmFyIGE9dHlwZW9mIHJlcXVpcmU9PVwiZnVuY3Rpb25cIiYmcmVxdWlyZTtpZighdSYmYSlyZXR1cm4gYShvLCEwKTtpZihpKXJldHVybiBpKG8sITApO3ZhciBmPW5ldyBFcnJvcihcIkNhbm5vdCBmaW5kIG1vZHVsZSAnXCIrbytcIidcIik7dGhyb3cgZi5jb2RlPVwiTU9EVUxFX05PVF9GT1VORFwiLGZ9dmFyIGw9bltvXT17ZXhwb3J0czp7fX07dFtvXVswXS5jYWxsKGwuZXhwb3J0cyxmdW5jdGlvbihlKXt2YXIgbj10W29dWzFdW2VdO3JldHVybiBzKG4/bjplKX0sbCxsLmV4cG9ydHMsZSx0LG4scil9cmV0dXJuIG5bb10uZXhwb3J0c312YXIgaT10eXBlb2YgcmVxdWlyZT09XCJmdW5jdGlvblwiJiZyZXF1aXJlO2Zvcih2YXIgbz0wO288ci5sZW5ndGg7bysrKXMocltvXSk7cmV0dXJuIHN9KSIsIm1vZHVsZS5leHBvcnRzID0gZnVuY3Rpb24gYXNzZXJ0KGNvbmRpdGlvbikge1xuICBpZiAoIWNvbmRpdGlvbikge1xuICAgIHRocm93IG5ldyBFcnJvcihcImFzc2VydGlvbiBmYWlsZWRcIik7XG4gIH1cbn1cbiIsInZhciBhc3NlcnQgPSByZXF1aXJlKFwiLi9hc3NlcnRcIik7XG5cbmZ1bmN0aW9uIExpbmFsZyhtZW1vcnkpIHtcbiAgZnVuY3Rpb24gQXNtTW9kdWxlKHN0ZGxpYiwgZm9yZWlnbiwgYnVmZmVyKSB7XG4gICAgXCJ1c2UgYXNtXCI7XG5cbiAgICB2YXIgaW11bCA9IHN0ZGxpYi5NYXRoLmltdWw7XG4gICAgdmFyIGZyb3VuZCA9IHN0ZGxpYi5NYXRoLmZyb3VuZDtcblxuICAgIHZhciBhcnIgPSBuZXcgc3RkbGliLkZsb2F0MzJBcnJheShidWZmZXIpO1xuXG4gICAgZnVuY3Rpb24gbWF0TXVsdChtYXQsIGluVmVjLCBvdXRWZWMsIG5Sb3dzLCBuQ29scykge1xuICAgICAgbWF0ID0gbWF0fDA7XG4gICAgICBpblZlYyA9IGluVmVjfDA7XG4gICAgICBvdXRWZWMgPSBvdXRWZWN8MDtcbiAgICAgIG5Sb3dzID0gblJvd3N8MDtcbiAgICAgIG5Db2xzID0gbkNvbHN8MDtcblxuICAgICAgdmFyIG1hdFB0ciA9IDAsIGluVmVjUHRyID0gMCwgb3V0VmVjUHRyID0gMDtcbiAgICAgIHZhciBtYXRFbmQgPSAwLCBpblZlY0VuZCA9IDAsIG91dFZlY0VuZCA9IDA7XG5cbiAgICAgIG1hdEVuZCA9IG1hdCArIChpbXVsKG5Sb3dzLCBuQ29scykgPDwgMil8MDtcbiAgICAgIGluVmVjRW5kID0gaW5WZWMgKyAobkNvbHMgPDwgMil8MDtcbiAgICAgIG91dFZlY0VuZCA9IG91dFZlYyArIChuUm93cyA8PCAyKXwwO1xuXG4gICAgICBtYXRQdHIgPSBtYXQ7XG4gICAgICBmb3IgKG91dFZlY1B0ciA9IG91dFZlYztcbiAgICAgICAgICAgKG91dFZlY1B0cnwwKSA8IChvdXRWZWNFbmR8MCk7XG4gICAgICAgICAgIG91dFZlY1B0ciA9IChvdXRWZWNQdHIgKyA0KXwwKSB7XG4gICAgICAgIGFycltvdXRWZWNQdHIgPj4gMl0gPSAwLjtcbiAgICAgICAgZm9yIChpblZlY1B0ciA9IGluVmVjO1xuICAgICAgICAgICAgIChpblZlY1B0cnwwKSA8IChpblZlY0VuZHwwKTtcbiAgICAgICAgICAgICBpblZlY1B0ciA9IChpblZlY1B0ciArIDQpfDAsIG1hdFB0ciA9IChtYXRQdHIgKyA0KXwwKSB7XG4gICAgICAgICAgYXJyW291dFZlY1B0ciA+PiAyXSA9IGFycltvdXRWZWNQdHIgPj4gMl0gK1xuICAgICAgICAgICAgZnJvdW5kKGFyclttYXRQdHIgPj4gMl0gKiBhcnJbaW5WZWNQdHIgPj4gMl0pO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuXG4gICAgcmV0dXJuIHsgbWF0TXVsdDogbWF0TXVsdCB9O1xuICB9XG5cbiAgdmFyIG1hdE11bHRBc20gPSBBc21Nb2R1bGUoZ2xvYmFsLCBudWxsLCBtZW1vcnkuYnVmZmVyKS5tYXRNdWx0O1xuXG4gIGZ1bmN0aW9uIG1hdE11bHQobWF0cml4LCBpblZlYywgb3V0VmVjKSB7XG4gICAgYXNzZXJ0KG1hdHJpeC5uQ29scyA9PSBpblZlYy5sZW5ndGgpO1xuICAgIGFzc2VydChtYXRyaXgublJvd3MgPT0gb3V0VmVjLmxlbmd0aCk7XG4gICAgYXNzZXJ0KG1hdHJpeC5idWZmZXIgPT0gbWVtb3J5LmJ1ZmZlcik7XG4gICAgYXNzZXJ0KGluVmVjLmJ1ZmZlciA9PSBtZW1vcnkuYnVmZmVyKTtcbiAgICBhc3NlcnQob3V0VmVjLmJ1ZmZlciA9PSBtZW1vcnkuYnVmZmVyKTtcblxuICAgIG1hdE11bHRBc20obWF0cml4LmJ5dGVPZmZzZXQsXG4gICAgICAgICAgICAgICBpblZlYy5ieXRlT2Zmc2V0LFxuICAgICAgICAgICAgICAgb3V0VmVjLmJ5dGVPZmZzZXQsXG4gICAgICAgICAgICAgICBtYXRyaXgublJvd3MsXG4gICAgICAgICAgICAgICBtYXRyaXgubkNvbHMpO1xuXG4gICAgcmV0dXJuIG91dFZlYztcbiAgfVxuXG4gIGZ1bmN0aW9uIG1hcDEoZnVuYykge1xuICAgIHJldHVybiBmdW5jdGlvbihpblZlYywgb3V0VmVjKSB7XG4gICAgICBpZiAoIW91dFZlYykge1xuICAgICAgICBvdXRWZWMgPSBtZW1vcnkubWFsbG9jKGluVmVjLmxlbmd0aCk7XG4gICAgICB9XG4gICAgICBhc3NlcnQoaW5WZWMubGVuZ3RoID09IG91dFZlYy5sZW5ndGgpO1xuICAgICAgZm9yICh2YXIgbiA9IDA7IG4gPCBpblZlYy5sZW5ndGg7IG4rKykge1xuICAgICAgICBvdXRWZWNbbl0gPSBmdW5jKGluVmVjW25dKTtcbiAgICAgIH1cbiAgICAgIHJldHVybiBvdXRWZWM7XG4gICAgfVxuICB9XG5cbiAgZnVuY3Rpb24gbWFwMihmdW5jKSB7XG4gICAgcmV0dXJuIGZ1bmN0aW9uKG9uZSwgdHdvLCBvdXQpIHtcbiAgICAgIGFzc2VydChvbmUubGVuZ3RoID09IHR3by5sZW5ndGgpO1xuICAgICAgaWYgKCFvdXQpIHtcbiAgICAgICAgb3V0ID0gbWVtb3J5Lm1hbGxvYyhvbmUubGVuZ3RoKTtcbiAgICAgIH1cbiAgICAgIGFzc2VydChvbmUubGVuZ3RoID09IG91dC5sZW5ndGgpO1xuICAgICAgZm9yICh2YXIgbiA9IDA7IG4gPCBvbmUubGVuZ3RoOyBuKyspIHtcbiAgICAgICAgb3V0W25dID0gZnVuYyhvbmVbbl0sIHR3b1tuXSk7XG4gICAgICB9XG4gICAgICByZXR1cm4gb3V0O1xuICAgIH1cbiAgfVxuXG4gIHZhciBhZGQgPSBtYXAyKCh4LCB5KSA9PiB4ICsgeSk7XG4gIHZhciBtdWx0ID0gbWFwMigoeCwgeSkgPT4geCAqIHkpO1xuXG4gIGZ1bmN0aW9uIG1ha2VBZmZpbmVUcmFuc2Zvcm1hdGlvbihsaW5lYXIsIHNoaWZ0KSB7XG4gICAgZnVuY3Rpb24gYWZmaW5lKGluVmVjLCBvdXRWZWMpIHtcbiAgICAgIGlmICghb3V0VmVjKSB7XG4gICAgICAgIG91dFZlYyA9IG1lbW9yeS5tYWxsb2MobGluZWFyLm5Sb3dzKTtcbiAgICAgIH1cbiAgICAgIGFzc2VydChpblZlYy5ieXRlT2Zmc2V0ICE9IG91dFZlYy5ieXRlT2Zmc2V0KTtcbiAgICAgIG1hdE11bHQobGluZWFyLCBpblZlYywgb3V0VmVjKTtcbiAgICAgIGFkZChvdXRWZWMsIHNoaWZ0LCBvdXRWZWMpO1xuICAgICAgcmV0dXJuIG91dFZlYztcbiAgICB9XG4gICAgYWZmaW5lLmluTGVuZ3RoID0gbGluZWFyLm5Db2xzO1xuICAgIGFmZmluZS5vdXRMZW5ndGggPSBsaW5lYXIublJvd3M7XG4gICAgYWZmaW5lLmxpbmVhciA9IGxpbmVhcjtcbiAgICBhZmZpbmUuc2hpZnQgPSBzaGlmdDtcbiAgICByZXR1cm4gYWZmaW5lO1xuICB9XG5cbiAgZnVuY3Rpb24gc2lnbW9pZChpblZlYywgb3V0VmVjKSB7XG4gICAgaWYgKCFvdXRWZWMpIHtcbiAgICAgIG91dFZlYyA9IG1lbW9yeS5tYWxsb2MoaW5WZWMubGVuZ3RoKTtcbiAgICB9XG4gICAgYXNzZXJ0KGluVmVjLmxlbmd0aCA9PSBvdXRWZWMubGVuZ3RoKTtcbiAgICBmb3IgKHZhciBuID0gMDsgbiA8IGluVmVjLmxlbmd0aDsgbisrKSB7XG4gICAgICBvdXRWZWNbbl0gPSAxIC8gKDEgKyBNYXRoLmV4cCgtaW5WZWNbbl0pKTtcbiAgICB9XG4gICAgcmV0dXJuIG91dFZlYztcbiAgfVxuXG4gIGZ1bmN0aW9uIHNjYWxhck11bHQoaW5WZWMsIHNjYWxhciwgb3V0VmVjKSB7XG4gICAgcmV0dXJuIChtYXAxKHggPT4gc2NhbGFyICogeCkoaW5WZWMsIG91dFZlYykpO1xuICB9XG5cbiAgcmV0dXJuIHtcbiAgICB2ZWNBZGRFbGVtczogYWRkLFxuICAgIHZlY011bHRFbGVtczogbXVsdCxcbiAgICBzY2FsYXJNdWx0OiBzY2FsYXJNdWx0LFxuICAgIC8vc2lnbW9pZDogbWFwMSh4ID0+IDEgLyAoMSArIE1hdGguZXhwKC14KSkpLFxuICAgIHNpZ21vaWQ6IHNpZ21vaWQsXG4gICAgdGFuaDogbWFwMShNYXRoLnRhbmgpLFxuICAgIGV4cDogbWFwMShNYXRoLmV4cCksXG4gICAgbG9nOiBtYXAxKE1hdGgubG9nKSxcbiAgICB6ZXJvOiBtYXAxKHggPT4gMCksXG4gICAgbWFrZUFmZmluZVRyYW5zZm9ybWF0aW9uOiBtYWtlQWZmaW5lVHJhbnNmb3JtYXRpb24sXG4gIH07XG59XG5tb2R1bGUuZXhwb3J0cyA9IExpbmFsZztcbiIsIi8vIFNlZSBodHRwczovL2dpdGh1Yi5jb20va2FycGF0aHkvY2hhci1ybm4vYmxvYi9tYXN0ZXIvbW9kZWwvTFNUTS5sdWFcbmZ1bmN0aW9uIExTVE0obWVtb3J5LCBsaW5hbGcsIHBhcmFtcykge1xuICBjb25zdCBleHBvcnRzID0ge307XG5cbiAgZnVuY3Rpb24gbWFrZVN0YXRlKCkge1xuICAgIHZhciBzdGF0ZSA9IG1lbW9yeS5tYWxsb2MocGFyYW1zLm5Ob2RlcyAqIHBhcmFtcy5uTGF5ZXJzICogMik7XG4gICAgcmV0dXJuIHJlc2V0U3RhdGUoc3RhdGUpO1xuICB9XG4gIGV4cG9ydHMubWFrZVN0YXRlID0gbWFrZVN0YXRlO1xuXG4gIGZ1bmN0aW9uIHJlc2V0U3RhdGUoc3RhdGUpIHtcbiAgICBsaW5hbGcuemVybyhzdGF0ZSwgc3RhdGUpO1xuICAgIHJldHVybiBzdGF0ZTtcbiAgfVxuICBleHBvcnRzLnJlc2V0U3RhdGUgPSByZXNldFN0YXRlO1xuXG4gIGZ1bmN0aW9uIGZvcndhcmQoaW5TdGF0ZSwgYnl0ZSwgb3V0U3RhdGUpIHtcbiAgICBtZW1vcnkucHVzaEZyYW1lKCk7XG4gICAgdmFyIGlucHV0ID0gYnl0ZVRvVmVjdG9yKGJ5dGUpO1xuICAgIGZvciAodmFyIG4gPSAwOyBuIDwgcGFyYW1zLm5MYXllcnM7IG4rKykge1xuICAgICAgaW5wdXQgPSBmb3J3YXJkTGF5ZXIoaW5kZXhTdGF0ZShpblN0YXRlLCAyICogbiksXG4gICAgICAgICAgICAgICAgICAgICAgICAgICBpbmRleFN0YXRlKGluU3RhdGUsIDIgKiBuICsgMSksXG4gICAgICAgICAgICAgICAgICAgICAgICAgICBpbnB1dCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIHBhcmFtcy5hZmZpbmVzWzIgKiBuXSxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIHBhcmFtcy5hZmZpbmVzWzIgKiBuICsgMV0sXG4gICAgICAgICAgICAgICAgICAgICAgICAgICBpbmRleFN0YXRlKG91dFN0YXRlLCAyICogbiksXG4gICAgICAgICAgICAgICAgICAgICAgICAgICBpbmRleFN0YXRlKG91dFN0YXRlLCAyICogbiArIDEpKTtcbiAgICB9XG4gICAgbWVtb3J5LnBvcEZyYW1lKCk7XG4gICAgcmV0dXJuIG91dFN0YXRlO1xuICB9XG4gIGV4cG9ydHMuZm9yd2FyZCA9IGZvcndhcmQ7XG5cbiAgZnVuY3Rpb24gcHJlZGljdChzdGF0ZSkge1xuICAgIG1lbW9yeS5wdXNoRnJhbWUoKTtcbiAgICB2YXIgdG9wSCA9IGluZGV4U3RhdGUoc3RhdGUsIDIgKiBwYXJhbXMubkxheWVycyAtIDEpO1xuICAgIHZhciBwcm9icyA9IHBhcmFtcy5hZmZpbmVzWzIgKiBwYXJhbXMubkxheWVyc10odG9wSCk7XG4gICAgcHJvYnMgPSBsaW5hbGcuZXhwKHByb2JzLCBwcm9icyk7XG4gICAgcHJvYnMgPSBub3JtYWxpemVBbmRFeHBvcnQocHJvYnMpO1xuICAgIG1lbW9yeS5wb3BGcmFtZSgpO1xuICAgIHJldHVybiBwcm9icztcbiAgfVxuICBleHBvcnRzLnByZWRpY3QgPSBwcmVkaWN0O1xuXG4gIGZ1bmN0aW9uIGZvcndhcmRMYXllcihwcmV2QywgcHJldkgsIHgsIGkyaCwgaDJoLCBuZXh0QywgbmV4dEgpIHtcbiAgICBtZW1vcnkucHVzaEZyYW1lKCk7XG5cbiAgICB2YXIgYWxsSW5wdXRTdW1zID0gbGluYWxnLnZlY0FkZEVsZW1zKGkyaCh4KSwgaDJoKHByZXZIKSk7XG5cbiAgICB2YXIgaW5HYXRlID0gbGluYWxnLnNpZ21vaWQoaW5kZXhTdGF0ZShhbGxJbnB1dFN1bXMsIDApKTtcbiAgICB2YXIgZm9yZ2V0R2F0ZSA9IGxpbmFsZy5zaWdtb2lkKGluZGV4U3RhdGUoYWxsSW5wdXRTdW1zLCAxKSk7XG4gICAgdmFyIG91dEdhdGUgPSBsaW5hbGcuc2lnbW9pZChpbmRleFN0YXRlKGFsbElucHV0U3VtcywgMikpO1xuICAgIHZhciBpblRyYW5zZm9ybSA9IGxpbmFsZy50YW5oKGluZGV4U3RhdGUoYWxsSW5wdXRTdW1zLCAzKSk7XG5cbiAgICBsaW5hbGcudmVjQWRkRWxlbXMobGluYWxnLnZlY011bHRFbGVtcyhmb3JnZXRHYXRlLCBwcmV2QyksXG4gICAgICAgICAgICAgICAgICAgICAgIGxpbmFsZy52ZWNNdWx0RWxlbXMoaW5HYXRlLCBpblRyYW5zZm9ybSksXG4gICAgICAgICAgICAgICAgICAgICAgIG5leHRDKTtcbiAgICBsaW5hbGcudmVjTXVsdEVsZW1zKG91dEdhdGUsIGxpbmFsZy50YW5oKG5leHRDKSwgbmV4dEgpO1xuXG4gICAgbWVtb3J5LnBvcEZyYW1lKCk7XG4gICAgLy9jb25zb2xlLmxvZyhuZXh0Qyk7XG4gICAgLy9jb25zb2xlLmxvZyhuZXh0SCk7XG4gICAgcmV0dXJuIG5leHRIO1xuICB9XG5cbiAgZnVuY3Rpb24gaW5kZXhTdGF0ZShzdGF0ZSwgbikge1xuICAgIHJldHVybiBzdGF0ZS5zdWJhcnJheShuICogcGFyYW1zLm5Ob2RlcywgKG4gKyAxKSAqIHBhcmFtcy5uTm9kZXMpO1xuICB9XG5cbiAgZnVuY3Rpb24gbm9ybWFsaXplQW5kRXhwb3J0KHByb2JzKSB7XG4gICAgdmFyIHN1bSA9IHByb2JzLnJlZHVjZSgoeCwgeSkgPT4geCArIHkpO1xuICAgIHZhciBvdXRQcm9icyA9IG5ldyBGbG9hdDMyQXJyYXkocHJvYnMubGVuZ3RoKTtcbiAgICByZXR1cm4gbGluYWxnLnNjYWxhck11bHQocHJvYnMsIDEgLyBzdW0sIG91dFByb2JzKTtcbiAgfVxuXG4gIGZ1bmN0aW9uIGJ5dGVUb0luZGV4KGJ5dGUpIHtcbiAgICByZXR1cm4gcGFyYW1zLnZvY2FiW2J5dGVdO1xuICB9XG4gIGV4cG9ydHMuYnl0ZVRvSW5kZXggPSBieXRlVG9JbmRleDtcblxuICBmdW5jdGlvbiBpbmRleFRvQnl0ZShpbmRleCkge1xuICAgIHJldHVybiBwYXJhbXMuaXZvY2FiW2luZGV4XTtcbiAgfVxuICBleHBvcnRzLmJ5dGVUb0luZGV4ID0gYnl0ZVRvSW5kZXg7XG5cbiAgZnVuY3Rpb24gYnl0ZVRvVmVjdG9yKGJ5dGUpIHtcbiAgICB2YXIgdmVjID0gbWVtb3J5Lm1hbGxvYyhwYXJhbXMuYWZmaW5lc1swXS5pbkxlbmd0aCk7XG4gICAgbGluYWxnLnplcm8odmVjLCB2ZWMpO1xuICAgIHZlY1tieXRlVG9JbmRleChieXRlKV0gPSAxLjtcbiAgICByZXR1cm4gdmVjO1xuICB9XG5cbiAgcmV0dXJuIGV4cG9ydHM7XG59XG5tb2R1bGUuZXhwb3J0cyA9IExTVE07XG4iLCJ2YXIgYXNzZXJ0ID0gcmVxdWlyZShcIi4vYXNzZXJ0XCIpO1xuXG5mdW5jdGlvbiBNZW1vcnkoYnVmZmVyLCBuZXh0KSB7XG4gIHRoaXMuYnVmZmVyID0gYnVmZmVyO1xuICB0aGlzLm5leHQgPSBuZXh0O1xuICB0aGlzLnN0YWNrID0gW107XG59XG5cbk1lbW9yeS5wcm90b3R5cGUubWFsbG9jID0gZnVuY3Rpb24oc2l6ZSkge1xuICB2YXIgYXJyID0gbmV3IEZsb2F0MzJBcnJheSh0aGlzLmJ1ZmZlciwgdGhpcy5uZXh0LCBzaXplKTtcbiAgdGhpcy5uZXh0ICs9IHNpemUgKiA0O1xuICBhc3NlcnQodGhpcy5uZXh0IDwgdGhpcy5idWZmZXIuYnl0ZUxlbmd0aCk7XG4gIHJldHVybiBhcnI7XG59XG5cbk1lbW9yeS5wcm90b3R5cGUucHVzaEZyYW1lID0gZnVuY3Rpb24oKSB7XG4gIHRoaXMuc3RhY2sucHVzaCh0aGlzLm5leHQpO1xufVxuXG5NZW1vcnkucHJvdG90eXBlLnBvcEZyYW1lID0gZnVuY3Rpb24gZnJlZShhcnJheSkge1xuICB0aGlzLm5leHQgPSB0aGlzLnN0YWNrLnBvcCgpO1xuICBhc3NlcnQodGhpcy5uZXh0ICE9PSB1bmRlZmluZWQpO1xufVxuXG5tb2R1bGUuZXhwb3J0cyA9IGZ1bmN0aW9uKGJ1ZmZlciwgbmV4dCkge1xuICByZXR1cm4gbmV3IE1lbW9yeShidWZmZXIsIG5leHQpO1xufVxuIiwibW9kdWxlLmV4cG9ydHMgPSBmdW5jdGlvbihtZW1vcnksIG1vZGVsKSB7XG4gIFwidXNlIHN0cmljdFwiXG5cbiAgcmV0dXJuIGZ1bmN0aW9uIHNjb3JlKGJ5dGVzKSB7XG4gICAgbWVtb3J5LnB1c2hGcmFtZSgpO1xuXG4gICAgdmFyIHN0YXRlID0gbW9kZWwubWFrZVN0YXRlKCk7XG4gICAgdmFyIGN1cnJlbnRTY29yZSA9IDA7XG4gICAgZm9yICh2YXIgbiA9IDA7IG4gPCBieXRlcy5sZW5ndGg7IG4rKykge1xuICAgICAgaWYgKG4gPiAwKSB7XG4gICAgICAgIHZhciBwcm9icyA9IG1vZGVsLnByZWRpY3Qoc3RhdGUpO1xuICAgICAgICBjdXJyZW50U2NvcmUgKz0gTWF0aC5sb2cocHJvYnNbbW9kZWwuYnl0ZVRvSW5kZXgoYnl0ZXNbbl0pXSk7XG4gICAgICB9XG4gICAgICBtb2RlbC5mb3J3YXJkKHN0YXRlLCBieXRlc1tuXSwgc3RhdGUpO1xuICAgIH1cblxuICAgIG1lbW9yeS5wb3BGcmFtZSgpO1xuICAgIHJldHVybiBjdXJyZW50U2NvcmU7XG4gIH1cbn1cbiIsInZhciBhc3NlcnQgPSByZXF1aXJlKFwiLi9hc3NlcnRcIik7XG5cbmZ1bmN0aW9uIHN0cmluZ1RvQnl0ZXMoc3RyKSB7XG4gIHJldHVybiB1bmVzY2FwZShlbmNvZGVVUklDb21wb25lbnQoc3RyKSk7XG59XG5cbmZ1bmN0aW9uIGJ5dGVzVG9TdHJpbmcoYnl0ZXMpIHtcbiAgcmV0dXJuIGRlY29kZVVSSUNvbXBvbmVudChlc2NhcGUoYnl0ZXMpKTtcbn1cblxuZnVuY3Rpb24gbG9hZChwYXRoLCBjYWxsYmFjaykge1xuICBmdW5jdGlvbiBnZXQodXJsLCB0eXBlLCBjYWxsYmFjaykge1xuICAgIHZhciByZXEgPSBuZXcgWE1MSHR0cFJlcXVlc3QoKTtcbiAgICByZXEub3BlbihcIkdFVFwiLCB1cmwsIHRydWUpO1xuICAgIHJlcS5yZXNwb25zZVR5cGUgPSB0eXBlO1xuICAgIHJlcS5vbmxvYWQgPSBmdW5jdGlvbigpIHtcbiAgICAgIGNhbGxiYWNrKHJlcS5yZXNwb25zZSk7XG4gICAgfVxuICAgIHJlcS5zZW5kKCk7XG4gICAgcmV0dXJuIHJlcTtcbiAgfVxuXG4gIGdldChwYXRoICsgXCIuZGF0P1wiLCBcImFycmF5YnVmZmVyXCIsIGZ1bmN0aW9uKGJ1ZmZlcikge1xuICAgIGdldChwYXRoICsgXCIuanNvblwiLCBcImpzb25cIiwgZnVuY3Rpb24obWV0YWRhdGEpIHtcbiAgICAgIGNvbnNvbGUubG9nKFwiR290IGRhdGFcIik7XG4gICAgICBpbml0KGJ1ZmZlciwgbWV0YWRhdGEpO1xuICAgICAgaWYgKGNhbGxiYWNrKSB7XG4gICAgICAgIGNhbGxiYWNrKCk7XG4gICAgICB9XG4gICAgfSk7XG4gIH0pO1xufVxuXG5mdW5jdGlvbiBpbml0KGJ1ZmZlciwgbWV0YWRhdGEpIHtcbiAgdmFyIG1lbW9yeSA9IHJlcXVpcmUoXCIuL21lbW9yeVwiKShidWZmZXIsIG1ldGFkYXRhLm5leHQpO1xuICB2YXIgbGluYWxnID0gcmVxdWlyZShcIi4vbGluYWxnXCIpKG1lbW9yeSk7XG5cbiAgdmFyIGFycmF5cyA9IFtdXG4gIGZvciAodmFyIG4gPSAwOyBuIDwgbWV0YWRhdGEuYXJyYXlzLmxlbmd0aDsgbisrKSB7XG4gICAgdmFyIGFycmF5ID1cbiAgICAgIG5ldyBGbG9hdDMyQXJyYXkoYnVmZmVyLCBtZXRhZGF0YS5hcnJheXNbbl0ub2Zmc2V0LCBtZXRhZGF0YS5hcnJheXNbbl0ubGVuZ3RoKTtcbiAgICB2YXIgZGltcyA9IG1ldGFkYXRhLmFycmF5c1tuXS5kaW07XG4gICAgaWYgKGRpbXMubGVuZ3RoID4gMSkge1xuICAgICAgYXJyYXkublJvd3MgPSBkaW1zWzBdO1xuICAgICAgYXJyYXkubkNvbHMgPSBkaW1zWzFdO1xuICAgIH1cbiAgICBhcnJheS5kaW1zID0gZGltcztcbiAgICBhcnJheXMucHVzaChhcnJheSk7XG4gIH1cblxuICB2YXIgYWZmaW5lcyA9IFtdO1xuICBmb3IgKHZhciBuID0gMDsgbiA8IGFycmF5cy5sZW5ndGggLyAyOyBuKyspIHtcbiAgICBhZmZpbmVzLnB1c2gobGluYWxnLm1ha2VBZmZpbmVUcmFuc2Zvcm1hdGlvbihhcnJheXNbMiAqIG5dLCBhcnJheXNbMiAqIG4gKyAxXSkpO1xuICB9XG5cbiAgdmFyIHBhcmFtcyA9IHt9O1xuICBwYXJhbXMuYWZmaW5lcyA9IGFmZmluZXM7XG4gIHBhcmFtcy5uTm9kZXMgPSBhZmZpbmVzWzBdLm91dExlbmd0aCAvIDQ7XG4gIHBhcmFtcy5uTGF5ZXJzID0gKGFmZmluZXMubGVuZ3RoIC0gMSkgLyAyO1xuXG4gIHBhcmFtcy52b2NhYiA9IHt9O1xuICBwYXJhbXMuaXZvY2FiID0ge307XG4gIGZvciAodmFyIG4gPSAwOyBuIDwgbWV0YWRhdGEudm9jYWIubGVuZ3RoOyBuKyspIHtcbiAgICBpZiAobWV0YWRhdGEudm9jYWJbbl0gIT09IG51bGwpIHtcbiAgICAgIHBhcmFtcy52b2NhYltTdHJpbmcuZnJvbUNoYXJDb2RlKG4pXSA9IG1ldGFkYXRhLnZvY2FiW25dO1xuICAgICAgcGFyYW1zLml2b2NhYlttZXRhZGF0YS52b2NhYltuXV0gPSBTdHJpbmcuZnJvbUNoYXJDb2RlKG4pO1xuICAgIH1cbiAgfVxuXG4gIHZhciBtb2RlbCA9IHJlcXVpcmUoXCIuL2xzdG1cIikobWVtb3J5LCBsaW5hbGcsIHBhcmFtcyk7XG5cbiAgdmFyIGQgPSBuZXcgRGF0ZSgpO1xuICBmb3IgKHZhciBuID0gMDsgbiA8IDEwMDsgbisrKSB7XG4gICAgbW9kZWwucHJlZGljdChtb2RlbC5mb3J3YXJkKG1vZGVsLm1ha2VTdGF0ZSgpLCBcImFcIiwgbW9kZWwubWFrZVN0YXRlKCkpKTtcbiAgfVxuICBjb25zb2xlLmxvZyhuZXcgRGF0ZSgpIC0gZCk7XG4gIGNvbnNvbGUubG9nKG1vZGVsLnByZWRpY3QobW9kZWwuZm9yd2FyZChtb2RlbC5tYWtlU3RhdGUoKSwgXCJhXCIsIG1vZGVsLm1ha2VTdGF0ZSgpKSkpO1xuICBjb25zb2xlLmxvZyhtb2RlbC5wcmVkaWN0KG1vZGVsLmZvcndhcmQobW9kZWwubWFrZVN0YXRlKCksIFwiYVwiLCBtb2RlbC5tYWtlU3RhdGUoKSkpKTtcblxuICBnbG9iYWwuc2NvcmUgPSByZXF1aXJlKFwiLi9zY29yZVwiKShtZW1vcnksIG1vZGVsKTtcblxuICB2YXIgZCA9IG5ldyBEYXRlKCk7XG4gIGZvciAodmFyIG4gPSAwOyBuIDwgMTA7IG4rKykge1xuICAgIGdsb2JhbC5zY29yZShcIiB0aGlzIGlzIHNvbWUgdGV4dCBmb28gYmFyIGJhelwiKTtcbiAgfVxuICBjb25zb2xlLmxvZyhuZXcgRGF0ZSgpIC0gZCk7XG5cbiAgcmV0dXJuIHsgbWVtb3J5OiBtZW1vcnksIG1vZGVsOiBtb2RlbCB9O1xufVxuXG4vL2xvYWQoXCJkYXRhL2xhcmdlXCIpO1xuXG5nbG9iYWwubG9hZCA9IGxvYWQ7XG4iXX0=
