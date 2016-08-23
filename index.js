"use strict";

/**
 * @module char-rnn.js
 * @description
 * A javaScript port of https://github.com/karpathy/char-rnn
  * ### Instructions
  *
  * ```
  * npm install char-rnn
  * ```
  *
  * ### Create a new Network:
  *
  * ```
  * const CharRnn = require("./char-rnn");
  * const Memory = require("./memory");
  * const Linalg = require("./linalg");
  * const Lstm = require("./lstm");
  *
  *
  * const networkMemory = new Memory(buffer: a Buffer, metadata: a json representation of the buffer);
  *
  * const linalg = new Linalg(networkMemory);
  *
  * const params = {affines, nNodes, nLayers, vocab, ivocab}
  * // ^^^ there are more steps here. see load.js file for more information.
  *
  * const model = new LSTM(linalg, params);
  * const myNetwork = new CharRNN(model);
  * ```
  *
 */
module.exports = require("./lib/load");
