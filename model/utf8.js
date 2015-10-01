function stringToBytes(str) {
  return unescape(encodeURIComponent(str));
}

function bytesToString(bytes) {
  return decodeURIComponent(escape(bytes));
}

function getSequenceLength(firstByte) {
  var x = firstByte.charCodeAt(0);
  if (x < 128) {
    return 1;
  }
  if (x >> 5 == 0b110) {
    return 2;
  }
  if (x >> 4 == 0b1110) {
    return 3;
  }
  if (x >> 3 == 0b11110) {
    return 4;
  }
  // Invalid first byte
  return 0;
}

exports.stringToBytes = stringToBytes;
exports.bytesToString = bytesToString;
exports.getSequenceLength = getSequenceLength;
