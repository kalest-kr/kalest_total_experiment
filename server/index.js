// http 모듈 불러오기
const http = require("http");
//fs 모듈 불러오기
const fs = require("fs");

const PORT = Number(process.env.PORT) || 3000;

const server = http.createServer((req, res) => {
    console.log("요청 발생");
    console.log(req.method);
    // 응답헤더
    //res.setHeader("Contaent - Type", 'text/plain');
    //res.write("Hellow, Node");
    //res.end();

    //res.setHeader("Content-type", "text/html")
    //const readStream = fs.createReadStream(__dirname+"/index.html", "utf8")
    //readStream.pipe(res);
    const {method, URL} = req;
        
    if(method === 'GET' && URL === '/home'){
        res.statuscode = 200
        res.end("HOME")
    }

    else if (method === 'GET' && URL === '/about'){
        res.statusCode= 200;
        res.end("ABOUT")
    }
    else {
        res.statusCode = 404;
        res.end("NOT FOUND")
    }
});

server.listen(PORT, () => { // waiting으로 서버사이드 반응을 대기 ctrl + c로 실행취소 가능
    console.log(`$(PORT)포트로 서버 실행`);
});
