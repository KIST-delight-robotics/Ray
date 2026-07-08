/**
 * RAY virtual Matter On/Off light (matter.js).
 *
 * A real Matter device — not a mock. It announces itself on the local network
 * (mDNS) and speaks the actual Matter protocol, so the Raspberry Pi controller
 * (chip-tool, via our ChipToolBackend) can commission it on-network and toggle
 * it. Every on/off is printed here so you can watch the "signal" arrive.
 *
 * Fixed test commissioning parameters below yield the well-known manual pairing
 * code **34970112332** (matches matter_platform_led/config.toml), so no copying
 * codes around is needed for local verification.
 *
 * Run:  node light.mjs                 (state persists in ./storage)
 *       node light.mjs --storage-clear (factory reset: forget commissioning)
 */

import { DeviceTypeId, Endpoint, ServerNode, VendorId } from "@matter/main";
import { OnOffLightDevice } from "@matter/main/devices";

// Standard Matter test values. passcode 20202021 + discriminator 3840
// => manual pairing code 34970112332 / QR "MT:-24J0AFN00KA0648G00".
const PASSCODE = 20202021;
const DISCRIMINATOR = 3840;
const PORT = 5540;

const server = await ServerNode.create({
    id: "ray-virtual-light",
    network: { port: PORT },
    commissioning: { passcode: PASSCODE, discriminator: DISCRIMINATOR },
    productDescription: {
        name: "RAY Virtual Light",
        deviceType: DeviceTypeId(OnOffLightDevice.deviceType),
    },
    basicInformation: {
        vendorName: "RAY",
        vendorId: VendorId(0xfff1),
        nodeLabel: "RAY Virtual Light",
        productName: "RAY Virtual Light",
        productLabel: "RAY Virtual Light",
        productId: 0x8000,
        serialNumber: "ray-virtual-light-0001",
        uniqueId: "ray-virtual-light",
    },
});

// A Matter node is a composition of endpoints. One On/Off light endpoint here.
const light = new Endpoint(OnOffLightDevice, { id: "onoff" });
await server.add(light);

// React to on/off changes — this is the "light bulb" reacting to the controller.
light.events.onOff.onOff$Changed.on(value => {
    console.log(`\n>>> LIGHT is now ${value ? "ON  💡" : "OFF ⚫"}\n`);
});

console.log("RAY virtual Matter light starting — pairing code / QR is printed below.");
console.log(`(manual pairing code: 34970112332, discriminator: ${DISCRIMINATOR}, port: ${PORT})\n`);

// run() announces the node on the network and prints the QR code automatically.
await server.run();
