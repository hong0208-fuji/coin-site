const bcrypt = require("bcryptjs");

function verifyCredentials(username, password) {
  const expectedUsername = process.env.ADMIN_USERNAME;
  const expectedHash = process.env.ADMIN_PASSWORD_HASH;

  if (!username || !password || !expectedUsername || !expectedHash) {
    return false;
  }
  if (username !== expectedUsername) {
    return false;
  }
  return bcrypt.compareSync(password, expectedHash);
}

module.exports = { verifyCredentials };
