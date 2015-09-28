var assert = require("assert");
var Promise = require("promise");

var utils = require("./utils");
var stringToBytes = utils.stringToBytes;
var bytesToString = utils.bytesToString;

function loadFromUrl(path, callback) {
  function get(url, type) {
    return new Promise(function(resolve, reject) {
      var req = new XMLHttpRequest();
      req.open("GET", url, true);
      req.responseType = type;
      req.onload = function() {
        resolve(req.response);
      };
      req.onerror = function() {
        reject();
      };
      req.send();
      return req;
    });
  }

  return Promise.all([get(path + ".dat", "arraybuffer"),
                      get(path + ".json", "json")])
    .then(function(res) {
      console.log("Got data " + path);
      return new Model(res[0], res[1]);
    });
}

function Model(buffer, metadata) {
  if (!(this instanceof Model)) {
    return new Model(buffer, metadata);
  }

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

  var model = require("./lstm")(linalg, params);
  for (var key in model) {
    this[key] = model[key];
  }

  this.memory = memory;
  this.score = require("./score")(model);
}

Model.stringToBytes = stringToBytes;
Model.bytesToString = bytesToString;
Model.loadFromUrl = loadFromUrl;

Model.prototype.getState = function(str, initialState) {
  var bytes = Model.stringToBytes(str);
  var state = initialState ?
    this.copyState(initialState) :
    this.makeState();
  for (var n = 0; n < bytes.length; n++) {
    this.forward(state, bytes[n], state);
  }
  return state;
}

module.exports = Model;
