// Bench test for the MD80 FDCAN link -- CANdle HAT (SPI) or USB dongle.
//
//   candle_bus_test ping [--bus spi|usb|uart] [--baud 1|2|5|8] [--device PATH]
//   candle_bus_test soak [--seconds N] [--expect N] [bus/baud/device as above]
//   candle_bus_test baud --id ID --to 1|2|5|8 [bus/baud/device as above]
//
// ping  -- list drives answering on the bus.
// soak  -- stability run: add every drive found, start candle's auto-update
//          loop and stream read-only for N seconds, reporting the achieved
//          update frequency once a second. Motors are never enabled. Exits
//          non-zero on a frequency collapse or missing drives -- this is the
//          flake detector for the USB->SPI migration.
// baud  -- reconfigure ONE drive's CAN baudrate (needed once per drive for
//          the HAT: SPI runs FDCAN at 8M, our USB setup used 1M). Reads the
//          drive's current CAN watchdog + termination first and writes them
//          back unchanged, then saves to flash.
//
// Defaults are the HAT's: --bus spi --baud 8 --device /dev/spidev0.0 (candle's
// own default device when empty). For the USB dongle use --bus usb --baud 1.

#include <atomic>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "candle.hpp"

namespace
{
std::atomic<bool> g_stop{false};
void on_sigint(int) { g_stop = true; }

struct Args
{
  std::string mode;
  mab::BusType_E bus = mab::BusType_E::SPI;
  int baud = 8;
  std::string device;  // empty -> candle's per-bus default (SPI: /dev/spidev0.0)
  int seconds = 60;
  int expect = 0;      // soak: fail if fewer drives found (0 = just report)
  int id = -1;         // baud: drive to reconfigure
  int to = 0;          // baud: new baudrate in Mbps
};

bool parse_baud(const std::string & s, int & out)
{
  if (s != "1" && s != "2" && s != "5" && s != "8") return false;
  out = std::stoi(s);
  return true;
}

bool parse(int argc, char ** argv, Args & a)
{
  if (argc < 2) return false;
  a.mode = argv[1];
  if (a.mode != "ping" && a.mode != "soak" && a.mode != "baud") return false;
  for (int i = 2; i < argc; ++i) {
    std::string k = argv[i];
    auto next = [&]() -> const char * {
      return (i + 1 < argc) ? argv[++i] : nullptr;
    };
    const char * v;
    if (k == "--bus" && (v = next())) {
      if (!strcmp(v, "spi")) a.bus = mab::BusType_E::SPI;
      else if (!strcmp(v, "usb")) a.bus = mab::BusType_E::USB;
      else if (!strcmp(v, "uart")) a.bus = mab::BusType_E::UART;
      else return false;
    } else if (k == "--baud" && (v = next())) {
      if (!parse_baud(v, a.baud)) return false;
    } else if (k == "--device" && (v = next())) {
      a.device = v;
    } else if (k == "--seconds" && (v = next())) {
      a.seconds = std::stoi(v);
    } else if (k == "--expect" && (v = next())) {
      a.expect = std::stoi(v);
    } else if (k == "--id" && (v = next())) {
      a.id = std::stoi(v);
    } else if (k == "--to" && (v = next())) {
      if (!parse_baud(v, a.to)) return false;
    } else {
      return false;
    }
  }
  if (a.mode == "baud" && (a.id < 0 || a.to == 0)) return false;
  return true;
}

int run_ping(mab::Candle & candle, const Args & a)
{
  auto ids = candle.ping(static_cast<mab::CANdleBaudrate_E>(a.baud));
  std::cout << "found " << ids.size() << " drive(s):";
  for (auto id : ids) std::cout << " " << id;
  std::cout << std::endl;
  return ids.empty() ? 1 : 0;
}

int run_soak(mab::Candle & candle, const Args & a)
{
  auto ids = candle.ping(static_cast<mab::CANdleBaudrate_E>(a.baud));
  std::cout << "soak: " << ids.size() << " drive(s) on the bus" << std::endl;
  if (a.expect > 0 && static_cast<int>(ids.size()) != a.expect) {
    std::cerr << "FAIL: expected " << a.expect << " drives, found "
              << ids.size() << std::endl;
    return 1;
  }
  if (ids.empty()) return 1;
  for (auto id : ids) {
    if (!candle.addMd80(id)) {
      std::cerr << "FAIL: addMd80(" << id << ")" << std::endl;
      return 1;
    }
  }
  if (!candle.begin()) {
    std::cerr << "FAIL: candle.begin()" << std::endl;
    return 1;
  }

  int min_freq = -1;
  int bad_seconds = 0;
  for (int t = 0; t < a.seconds && !g_stop; ++t) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    const int freq = candle.getActualCommunicationFrequency();
    if (min_freq < 0 || freq < min_freq) min_freq = freq;
    // The update loop nominally runs at ~100 Hz regardless of bus; half of
    // that for a whole second means the link is choking.
    const bool bad = freq < 50;
    bad_seconds += bad;
    std::cout << "t=" << (t + 1) << "s freq=" << freq << "Hz pos[";
    for (auto & md : candle.md80s) std::cout << " " << md.getPosition();
    std::cout << " ]" << (bad ? "  <-- LOW" : "") << std::endl;
  }
  candle.end();

  std::cout << "soak done: min freq " << min_freq << " Hz, low seconds "
            << bad_seconds << std::endl;
  if (bad_seconds > 0) {
    std::cerr << "FAIL: link choked for " << bad_seconds << " s" << std::endl;
    return 1;
  }
  std::cout << "PASS" << std::endl;
  return 0;
}

int run_baud(mab::Candle & candle, const Args & a)
{
  const uint16_t id = static_cast<uint16_t>(a.id);
  // Preserve the drive's watchdog/termination -- configMd80Can writes all
  // four CAN registers at once, so read the two we do not want to change.
  uint16_t watchdog = 0;
  uint8_t termination = 0;
  if (!candle.readMd80Register(id, mab::Md80Reg_E::canWatchdog, watchdog) ||
      !candle.readMd80Register(id, mab::Md80Reg_E::canTermination, termination)) {
    std::cerr << "FAIL: could not read current CAN config of drive " << id
              << " (is it on this bus/baud?)" << std::endl;
    return 1;
  }
  std::cout << "drive " << id << ": watchdog " << watchdog << " ms, termination "
            << int(termination) << " -- switching to " << a.to << "M" << std::endl;
  if (!candle.configMd80Can(id, id, static_cast<mab::CANdleBaudrate_E>(a.to),
                            watchdog, termination != 0)) {
    std::cerr << "FAIL: configMd80Can" << std::endl;
    return 1;
  }
  // The drive switches to the new baudrate immediately, so the save command
  // must be sent at the NEW rate -- retune the CANdle side first, or the save
  // silently never reaches the drive (baud changed in RAM, lost on power cycle).
  if (a.to != a.baud &&
      !candle.configCandleBaudrate(static_cast<mab::CANdleBaudrate_E>(a.to))) {
    std::cerr << "FAIL: configCandleBaudrate(" << a.to
              << ") -- baud changed in RAM but NOT saved to flash" << std::endl;
    return 1;
  }
  if (!candle.configMd80Save(id)) {
    std::cerr << "FAIL: configMd80Save (baud changed but NOT saved to flash)"
              << std::endl;
    return 1;
  }
  std::cout << "PASS: drive " << id << " now at " << a.to
            << "M (saved to flash)" << std::endl;
  return 0;
}
}  // namespace

int main(int argc, char ** argv)
{
  Args a;
  if (!parse(argc, argv, a)) {
    std::cerr <<
      "usage: candle_bus_test ping|soak|baud [--bus spi|usb|uart] [--baud 1|2|5|8]\n"
      "                       [--device PATH] [--seconds N] [--expect N]\n"
      "                       [--id ID --to 1|2|5|8]\n";
    return 2;
  }
  std::signal(SIGINT, on_sigint);
  try {
    mab::Candle candle(static_cast<mab::CANdleBaudrate_E>(a.baud), true, a.bus,
                       a.device);
    if (a.mode == "ping") return run_ping(candle, a);
    if (a.mode == "soak") return run_soak(candle, a);
    return run_baud(candle, a);
  } catch (const char * e) {  // candle throws const char* on bus errors
    std::cerr << "FAIL: " << e << std::endl;
    return 1;
  } catch (const std::exception & e) {
    std::cerr << "FAIL: " << e.what() << std::endl;
    return 1;
  }
}
