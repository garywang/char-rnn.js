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
