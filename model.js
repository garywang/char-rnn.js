var assert = require("./assert");

function Model(internalModel) {
  function State(wrappedState) {
    this.wrappedState_ = wrappedState;
    this.freed = false;
  }

  State.prototype.copy = function() {
    var copy = makeState();
    for (var n = 0; n < copy.length; n++) {
      copy.wrappedState_[n] = this.wrappedState_[n];
    }
    return copy;
  }

  State.prototype.step = function(byte) {
    internalModel.forward(this.wrappedState_, byte, this.wrappedState_);
  }

  State.prototype.getPredictions = function() {
    return internalModel.predict(this.wrappedState_);
  }

  var freedStates = [];
  function makeState() {
    if (freedStates.length > 0) {
      var state = freedStates.pop();
      internalModel.resetState(state.wrappedState_);
      return state;
    } else {
      return new State(internalModel.makeState());
    }
  }

  function freeState(state) {
    freedStates.push(state);
  }

  function byteToIndex(byte) {
    return internalModel.byteToIndex[byte];
  }

  return { makeState: makeState, freeState: freeState, byteToIndex: byteToIndex };
}
module.exports = Model;
