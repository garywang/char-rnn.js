"use strict";

var assert = require("assert");

var loadFromUrl = require("./load").loadFromUrl;
var utils = require("./utils");
var stringToBytes = utils.stringToBytes;
var bytesToString = utils.bytesToString;

function CharRnn(model) {
  if (!(this instanceof CharRnn)) {
    return new CharRnn(model);
  }

  this._model = model;
}

CharRnn.prototype.getState = function(str, initialState) {
  if (!str) {
    str = "";
  }
  var bytes = stringToBytes(str);
  var state = initialState ?
    this._model.copyState(initialState) :
    this._model.makeState();

  for (var n = 0; n < bytes.length; n++) {
    this._model.forward(state, bytes[n], state);
  }

  return state;
}

CharRnn.prototype.score = function(str, initialState) {
  var bytes = stringToBytes(str);
  var state = initialState ?
    this._model.copyState(initialState) :
    this._model.makeState();

  var currentScore = 0;
  for (var n = 0; n < bytes.length; n++) {
    var probs = this._model.predict(state);
    currentScore += Math.log(probs[this._model.byteToIndex(bytes[n])]);
    console.log(currentScore);
    if (n < bytes.length - 1) {
      this._model.forward(state, bytes[n], state);
    }
  }

  return currentScore;
}

CharRnn.prototype.sample = function(state) {
  var probs = this._model.predict(state);
  var x = Math.random();
  for (var n = 0; n < probs.length; n++) {
    x -= probs[n];
    if (x < 0) {
      var byte = this._model.indexToByte(n);
      return byte;
    }
  }
  assert(false);
}

exports.loadFromUrl = function(url) {
  return loadFromUrl(url)
    .then(function(model) {
      return new CharRnn(model);
    });
};
