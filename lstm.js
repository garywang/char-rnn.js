// See https://github.com/karpathy/char-rnn/blob/master/model/LSTM.lua
function LSTM(memory, linalg, params) {
  const module = {};

  function makeState() {
    var state = memory.malloc(params.nNodes * params.nLayers * 2);
    linalg.zero(state, state);
    return state;
  }
  module.makeState = makeState;

  function makeProbs() {
    return memory.malloc(params.affines[params.affines.length - 1].outLength);
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
    var vec = memory.malloc(params.affines[0].inLength);
    vec[byteToIndex(byte)] = 1.;
    return vec;
  }

  return module;
}
