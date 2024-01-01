import jittor

if __name__ == "__main__":
    if jittor.has_cuda == 1:
        jittor.flags.use_cuda = 1
    print(jittor.compiler.has_cuda)
    print(jittor.flags.use_cuda)