const bcrypt = require("bcryptjs");

const password = process.argv[2];

if (!password) {
  console.error("사용법: npm run hash-password -- <비밀번호>");
  process.exit(1);
}

console.log(bcrypt.hashSync(password, 10));
