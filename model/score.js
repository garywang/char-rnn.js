var utils = require("./utils");

module.exports = function(memory, model) {
  "use strict"

  return function score(str) {
    memory.pushFrame();

    var bytes = utils.stringToBytes(str);

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
