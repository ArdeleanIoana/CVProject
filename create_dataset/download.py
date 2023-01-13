import imageio
import urllib.request
class GifDownloader:
    # destinationFolder : String - path of the folder where the gifs will be saved
    # source : String - name of the file where the urls are stored
    # nameTemp: String - template for the names of saved gifs , eq sample -> sample1.gif
    def __init__(self, destinationFolder, source, nameTemp):
        self.destionationFolder = destinationFolder
        self.source = source
        self.nameTemp = nameTemp
    def downloadGif(self, url, name):
        imdata = urllib.request.urlopen(url).read()
        imbytes = bytearray(imdata)
        open(self.destionationFolder + "\\" + name, "wb+").write(imdata)
        print("gif downloaded")

    def downloadNGifs(self, count):
        f = open(self.source,"r")
        i = 0
        for x in f:
            if i < count:
                self.downloadGif(x,self.nameTemp + str(i) + ".gif")
            else:
                break
            i = i + 1
        f.close()
