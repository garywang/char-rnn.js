var assert = require("assert");

/**
 * @external Buffer
 * @see https://nodejs.org/api/buffer.html#buffer_class_method_buffer_from_array
 */

/**
 * Memory Object
 * @class
 * @param {Buffer}   buffer [description]
 * @param {number} next   [description]
 */
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
