import enum
from s2clientprotocol import sc2api_pb2 as sc_pb

class Camp(enum.IntEnum):
  RED = "1"
  BLUE = "2"

difficulties = {
  "1": sc_pb.VeryEasy,
  "2": sc_pb.Easy,
  "3": sc_pb.Medium,
  "4": sc_pb.MediumHard,
  "5": sc_pb.Hard,
  "6": sc_pb.Harder,
  "7": sc_pb.VeryHard,
  "8": sc_pb.CheatVision,
  "9": sc_pb.CheatMoney,
  "A": sc_pb.CheatInsane,
}