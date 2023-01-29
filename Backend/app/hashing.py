from passlib.context import CryptContext

pwd_context = CryptContext(schemes = ["bcrypt"], deprecated = "auto")

# class hash to convert the password into its hash value
class Hash():
    def bcrypt(password:str):
        return pwd_context.hash(password)

    def verify(hashed, normal):
        return pwd_context.verify(normal, hashed)