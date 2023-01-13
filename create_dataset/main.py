from download import GifDownloader

from corrupter import Corrupter
PCPathToProject = "D:\\faculta\info\\an3\\sem1\\ComputerVision\\proiect\\project\\"

def download(count):
    downloader = GifDownloader(PCPathToProject + "dataset\\gifSamples",
                               PCPathToProject + "sampleGifURL.txt",
                               "basic")
    downloader.downloadNGifs(count)

def main():
    #download(500)
    x = Corrupter("basic",PCPathToProject + "dataset\\gifSamples",PCPathToProject + "dataset\\corruptedGifs")
    x.corruptNGifs(500)

if __name__ == '__main__':
    main()
