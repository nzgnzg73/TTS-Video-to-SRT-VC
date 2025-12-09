import logging

class Subtitle():
    def __init__(self,ext="srt"):
        sub_dict = {
            "srt":{
                "coma": ",",
                "header": "",
                "format": lambda i,start,end,text : f"{i + 1}\n{self.timeformat(start)} --> {self.timeformat(end if end != None else start)}\n{text}\n\n",
            },
            "vtt":{
                "coma": ".",
                "header": "WebVTT\n\n",
                "format": lambda i,start,end,text : f"{self.timeformat(start)} --> {self.timeformat(end if end != None else start)}\n{text}\n\n",
            },
            "txt":{
                "coma": "",
                "header": "",
                "format": lambda i,start,end,text : f"{text}\n",
            },
        }

        self.ext = ext
        self.coma = sub_dict[ext]["coma"]
        self.header = sub_dict[ext]["header"]
        self.format = sub_dict[ext]["format"]

    def timeformat(self,time):
        hours = time // 3600
        minutes = (time - hours * 3600) // 60
        seconds = time - hours * 3600 - minutes * 60
        milliseconds = (time - int(time)) * 1000
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}{self.coma}{int(milliseconds):03d}"
    
    def get_subtitle(self,segments, progress):
        output = self.header
        # for i, segment in enumerate(segments):
        for i, segment in enumerate(progress.tqdm(segments, desc="Whisper working...")):
            text = segment.text
            if text.startswith(' '):
                text = text[1:]
            try:
                result = self.format(i,segment.start, segment.end, text)
                output += result
                # logging.info(result)
            except Exception as e:
                logging.error(e,segment)
        return output
    
    def write_subtitle(self, segments, output_file, model, progress):
        # output_file = output_file.split('.')[0]
        model = model.replace("/","_")
        output_file += f".({model})."+self.ext
        subtitle = self.get_subtitle(segments,progress)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(subtitle)
        return output_file