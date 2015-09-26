var assert = require("assert");

const CONTEXT_LENGTH = 50;

module.exports = function Checker(smallModel, largeModel) {
  "use strict";

  function compare(contextBefore, contextAfter, original, replacement) {
    function compareWithModel(model) {
      model.memory.pushFrame();
      var state = model.getState(contextBefore);
      var diff = model.score(replacement + contextAfter, state) -
        model.score(original + contextAfter, state);
      var priorDiff = model.score(" " + replacement + " ") -
        model.score(" " + original + " ");
      if (priorDiff > 0) {
        diff -= priorDiff;
      }
      model.memory.popFrame();
      return diff;
    }
    var smallDiff = compareWithModel(smallModel);
    if (smallDiff < -1.) {
      return smallDiff;
    }
    console.log("large");
    return compareWithModel(largeModel);
  }

  function setsToReplacementFunc(sets) {
    function capitalize(s) {
      return s[0].toUpperCase() + s.slice(1);
    }

    var replacements = {};
    for (var i = 0; i < sets.length; i++) {
      for (var j = 0; j < sets[i].length; j++) {
        var one = sets[i][j];
        replacements[one] = [];
        replacements[capitalize(one)] = [];
        for (var k = 0; k < sets[i].length; k++) {
          if (j != k) {
            var two = sets[i][k];
            replacements[one].push(two);
            replacements[capitalize(one)].push(two);
          }
        }
      }
    }

    function setsReplacementFunc(token) {
      if (replacements[token]) {
        return replacements[token];
      }
      return [];
    }

    return setsReplacementFunc;
  }

  var defaultReplacementFunc = setsToReplacementFunc([
    ["its", "it's"],
    ["his", "he's"],
    ["there", "their", "they're"],
  ]);

  const TOKEN_RE = / |\n|,|\.|"|\(|\)|:|\?|!|\//g;
  function tokenize(str) {
    var tokens = [];
    var match;
    var lastIndex = 0;
    while ((match = TOKEN_RE.exec(str)) !== null) {
      if (match.index > lastIndex) {
        tokens.push(str.slice(lastIndex, match.index));
      }
      tokens.push(match[0]);
      lastIndex = match.index + match[0].length;
    }
    if (lastIndex < str.length) {
      tokens.push(str.slice(lastIndex));
    }
    return tokens;
  }

  function check(str, replacementFunc) {
    if (!replacementFunc) {
      replacementFunc = defaultReplacementFunc;
    }

    str = "\n" + str + "\n";
    var tokens = tokenize(str);
    var newStr = "";
    var strIndex = 0;
    for (var token of tokens) {
      var contextBefore = newStr.slice(-CONTEXT_LENGTH);
      var original = token;
      strIndex += token.length;
      var contextAfter = str.substr(strIndex, CONTEXT_LENGTH);

      var bestScore = 1.;
      var best = original;
      for (var replacement of replacementFunc(token)) {
        console.log(replacement);
        var score = compare(contextBefore, contextAfter, original, replacement);
        console.log(score);
        if (score > bestScore) {
          bestScore = score;
          best = replacement;
        }
      }
      newStr += best;
    }
    return newStr.slice(1, -1);
  }

  return { check: check };
}
