import ConfigParser


def parse(configfilename):
    # if isinstance(configfile, file):
    #     text = configfile.read()
    # else:
    #     with open(configfile) as f:
    #         text = f.read()
    parser = ConfigParser.SafeConfigParser()
    parser.read(configfilename)

    configdict = {kat: {opt: val for opt, val in parser.items(kat)} for kat in parser.sections()}
    return configdict


if __name__ == "__main__":
    res = parse("sample.config")
    temp = 0
