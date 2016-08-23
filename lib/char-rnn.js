"use strict";


var assert = require("assert");

var utf8 = require("./utf8");

/**
 * The CharRNN constructor. Instantiates an object that represents the neural network.
 * @class
 * @param {LSTM} model is aliased by model.
 */
function CharRnn(model) {
  if (!(this instanceof CharRnn)) {
    return new CharRnn(model);
  }

  this._model = model;
}

/**
 * @description Gets the network state. Calls {@link LSTM.prototype.makeState} or LSTM.prototype.copyState, which returns a {@link Vector}.
 * @access public
 * @alias CharRnn.prototype.getState
 * @memberof! CharRnn
 * @instance
 * @summary Get the Network state
 */
CharRnn.prototype.getState = function(str, initialState) {
  var model = this._model;

  if (!str) {
    str = "";
  }
  var bytes = utf8.stringToBytes(str);
  var state = initialState ?
    model.copyState(initialState) :
    model.makeState();

  for (var n = 0; n < bytes.length; n++) {
    model.forward(state, bytes[n], state);
  }

  return state;
}

/**
 * A prototype method, CharRnn#score
 * @alias CharRnn.prototype.score
 * @memberof! CharRnn
 * @instance
 */
CharRnn.prototype.score = function(str, initialState) {
  var model = this._model;

  var bytes = utf8.stringToBytes(str);
  var state = initialState ?
    model.copyState(initialState) :
    model.makeState();

  var currentScore = 0;
  for (var n = 0; n < bytes.length; n++) {
    var probs = model.predict(state);
    currentScore += Math.log(probs[model.byteToIndex(bytes[n])]);
    if (n < bytes.length - 1) {
      model.forward(state, bytes[n], state);
    }
  }

  return currentScore;
}

/**
 * A prototype method, CharRnn#sample
 * @alias CharRnn.prototype.sample
 * @memberof! CharRnn
 * @instance
 */
CharRnn.prototype.sample = function(state) {
  var model = this._model;

  function sampleByte(state) {
    var probs = model.predict(state);
    var x = Math.random();
    for (var n = 0; n < probs.length; n++) {
      x -= probs[n];
      if (x < 0) {
        return model.indexToByte(n);
      }
    }
    assert(false);
  }

  var bytes = sampleByte(state);
  var length = utf8.getSequenceLength(bytes);
  if (length == 0) {
    return "\x00";
  }
  if (length == 1) {
    return bytes;
  }
  state = model.copyState(state);
  for (var n = 1; n < length; n++) {
    model.forward(state, bytes[bytes.length - 1], state);
    bytes += sampleByte(state);
  }
  try {
    return utf8.bytesToString(bytes);
  } catch (e) {
    console.log(e);
    return "\x00";
  }
}

module.exports = CharRnn;
