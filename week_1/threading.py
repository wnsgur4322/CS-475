import os

if __name__ is "__main__":
    for t in [1, 2, 4, 6, 8]:
        for s in [2, 4, 6, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 3072, 4096]:
            print("NUMS=%d" % s)
            cmd = "g++ -DNUMS=%d -DNUMT=%d project_0.cpp -o prog -lm -fopenmp" % (s,t)
            os.system(cmd)
            cmd = "./prog"
            os.system(cmd)