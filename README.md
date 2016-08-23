# char-rnn.js

A javaScript port of <https://github.com/karpathy/char-rnn>

### Instructions

    npm install char-rnn

### Create a new Network:

    const CharRnn = require("./char-rnn");
    const Memory = require("./memory");
    const Linalg = require("./linalg");
    const Lstm = require("./lstm");


    const networkMemory = new Memory(buffer: a Buffer, metadata: a json representation of the buffer);

    const linalg = new Linalg(networkMemory);

    const params = {affines, nNodes, nLayers, vocab, ivocab}
    // ^^^ there are more steps here. see load.js file for more information.

    const model = new LSTM(linalg, params);
    const myNetwork = new CharRNN(model)

# CharRnn

The CharRNN constructor. Instantiates an object that represents the neural network.

**Parameters**

-   `model` **[LSTM](#lstm)** is aliased by model.

# CharRnn.prototype.getState

Gets the network state. Calls [LSTM.prototype.makeState](LSTM.prototype.makeState) or LSTM.prototype.copyState, which returns a [Vector](Vector).

**Parameters**

-   `str`
-   `initialState`

# CharRnn.prototype.score

A prototype method, CharRnn#score

**Parameters**

-   `str`
-   `initialState`

# CharRnn.prototype.sample

A prototype method, CharRnn#sample

**Parameters**

-   `state`

# Vector

Vector is instantiated using a Float32Array.

# Linalg

Object to perform Linear Algebra

**Parameters**

-   `memory` **[Memory](#memory)** object.

# load

takes a dat and json file with the same filename, and returns an instance of the CharRnn Model.

**Parameters**

-   `buffer` **[Object](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Object)** [description]
-   `metadata` **[Object](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Object)** [description]

Returns **[CharRnn](#charrnn)** [description]

# LSTM

Returns an LSTM Object. See [documentation](https://github.com/karpathy/char-rnn/blob/master/model/LSTM.lua). More about long short-term memory on [Wikipedia](https://en.wikipedia.org/wiki/Long_short-term_memory).

**Parameters**

-   `linalg` **[Linalg](#linalg)** Linear Algebra Object
-   `params` **[Object](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Object)**

Returns **[Object](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Object)**

# Memory

Memory Object

**Parameters**

-   `buffer` **[Buffer](https://nodejs.org/api/buffer.html)** [description]
-   `next` **[number](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Number)** [description]
