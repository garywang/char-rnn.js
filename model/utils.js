function stringToBytes(str) {
  return unescape(encodeURIComponent(str));
}

function bytesToString(bytes) {
  return decodeURIComponent(escape(bytes));
}

exports.stringToBytes = stringToBytes;
exports.bytesToString = bytesToString;
