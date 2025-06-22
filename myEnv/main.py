from fastapi import FastAPI

app = FastAPI()


'''
this function is in charge of handling get requests tha tgo to the / path
Other examples are:
@app.post()
@app.put()
@app.delete()
'''
@app.get("/")
async def root():
    return {"message": "Hello World"}