/**
 * Returns an LSTM Object. See {@link  https://github.com/karpathy/char-rnn/blob/master/model/LSTM.lua|documentation}. More about long short-term memory on {@link https://en.wikipedia.org/wiki/Long_short-term_memory|Wikipedia}.
 * @class
 * @param {Linalg} linalg Linear Algebra Object
 * @param {Object} params
 * @returns {Object}
 */
function LSTM(linalg, params) {
  const Vector = linalg.Vector;


  function makeState() {
    return new Vector(params.nNodes * params.nLayers * 2);
  }

  function resetState(state) {
    linalg.zero(state, state);
    return state;
  }

  function copyState(state) {
    return new Vector(state);
  }

  function forward(inState, byte, outState) {
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
    return outState;
  }

  function predict(state) {
    var topH = indexState(state, 2 * params.nLayers - 1);
    var probs = params.affines[2 * params.nLayers](topH);
    linalg.exp(probs, probs);
    normalize(probs, probs);
    return probs;
  }

  function forwardLayer(prevC, prevH, x, i2h, h2h, nextC, nextH) {
    var allInputSums = linalg.vecAddElems(i2h(x), h2h(prevH));

    var inGate = linalg.sigmoid(indexState(allInputSums, 0));
    var forgetGate = linalg.sigmoid(indexState(allInputSums, 1));
    var outGate = linalg.sigmoid(indexState(allInputSums, 2));
    var inTransform = linalg.tanh(indexState(allInputSums, 3));

    linalg.vecAddElems(linalg.vecMultElems(forgetGate, prevC),
                       linalg.vecMultElems(inGate, inTransform),
                       nextC);
    linalg.vecMultElems(outGate, linalg.tanh(nextC), nextH);

    //console.log(nextC);
    //console.log(nextH);
    return nextH;
  }

  function indexState(state, n) {
    return state.subarray(n * params.nNodes, (n + 1) * params.nNodes);
  }

  function normalize(inVec, outVec) {
    var sum = inVec.reduce((x, y) => x + y);
    return linalg.scalarMult(inVec, 1 / sum, outVec);
  }

  function byteToIndex(byte) {
    return params.vocab[byte];
  }

  function indexToByte(index) {
    return params.ivocab[index];
  }

  function byteToVector(byte) {
    var vec = new Vector(params.affines[0].inLength);
    vec[byteToIndex(byte)] = 1.;
    return vec;
  }

  return {
    makeState: makeState,
    copyState: copyState,
    resetState: resetState,
    forward: forward,
    predict: predict,
    indexToByte: indexToByte,
    byteToIndex: byteToIndex,
  };
}
module.exports = LSTM;
