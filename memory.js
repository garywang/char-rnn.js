var assert = require("./assert");

function Memory(buffer, next) {
  this.buffer = buffer;
  this.next = next;
  this.stack = [];
}

Memory.prototype.malloc = function(size) {
  var arr = new Float32Array(this.buffer, this.next, size);
  this.next += size * 4;
  assert(this.next < this.buffer.byteLength);
  return arr;
}

Memory.prototype.pushFrame = function() {
  this.stack.push(this.next);
}

Memory.prototype.popFrame = function free(array) {
  this.next = this.stack.pop();
  assert(this.next !== undefined);
}

module.exports = function(buffer, next) {
  return new Memory(buffer, next);
}
