//express로 서버 만들기 

//모듈 임포트
const express = require("express");

//서버 생성
const app = express();

//포트 설정
const PORT = Number(process.env.PORT) || 3000;

//라우팅 설정
app.get("/", (req, res) => {
    res.status(200);
    res.send("HELLO NODE");
})

app.get("/about", (req, res) => {
    res.status(200);
    res.send("about");
}).post("/about", (req, res) => { // 이것도 가능
    res.status(200);
    res.send("about");
})

app.post("/", (req, res) => {
    res.status(200);
    res.send("HELLO NODE");
})

app.get("/:id", (req, res) => { // '/'뒤를 변수로 받음
    const userID = req.params.id;
    res.status(200);
    res.send(`${userID} 사용자에 따른 정보 요청`);
})

//서버 실행
app.listen(PORT, () => {
    console.log(`${PORT} 포트로 서버 실행`)
})