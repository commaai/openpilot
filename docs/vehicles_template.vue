<template>
<div class="supported-cars">
<section class="cover-image"></section>
<section>
    <div class="container">
        <div class="title-section">
            <h1 class="font-Monument">openpilot supports 150+ vehicles</h1>
            <p class="font-large">A supported vehicle is one that just works when you install a comma device. <!--Supported cars are arranged into three tiers: Gold, Silver, and Bronze.--> Every car performs differently with openpilot, but all supported cars should provide a better experience than any stock system.</p>
        </div>
    </div>
</section>

<section class="mobile-hide"> <!-- How we rate cars - section start -->
    <div class="container">
        <p class="font-JetBrains">HOW WE RATE THE CARS</p>

        <div class="rating-row">
            <div class="rating-row-item rating-row-item-primary">
                <img src="/supported-cars/icon-longitudinal.svg" alt="">
                <p>Gas & Brakes</p>
            </div>
            <div class="rating-row-item rating-row-item-secondary">
                <p>openpilot Adaptive Cruise Control (ACC)</p>
                <div class="rating-item">
                    {{Star.FULL.html_icon}}
                    <p>openpilot is able to control the gas and brakes.</p>
                </div>
                <div class="rating-item">
                    {{Star.HALF.html_icon}}
                    <p>openpilot is able to control the gas and brakes with some restrictions.</p>
                </div>
                <div class="rating-item">
                    {{Star.EMPTY.html_icon}}
                    <p>The gas and brakes are controlled by the car's stock Adaptive Cruise Control (ACC) system.</p>
                </div>
            </div>
            <div class="rating-row-item rating-row-item-secondary">
                <p>Stop and Go</p>
                <div class="rating-item">
                    {{Star.FULL.html_icon}}
                    <p>Adaptive Cruise Control (ACC) operates down to 0 mph.</p>
                </div>
                <div class="rating-item">
                    {{Star.EMPTY.html_icon}}
                    <p>Adaptive Cruise Control (ACC) available only above certain speeds. See your car's manual for the minimum speed.</p>
                </div>
            </div>
        </div>

        <div class="rating-row">
            <div class="rating-row-item rating-row-item-primary">
                <img src="/supported-cars/icon-steering.svg" alt="">
                <p>Steering</p>
            </div>
            <div class="rating-row-item rating-row-item-secondary">
                <p>Steer to 0</p>
                <div class="rating-item">
                    {{Star.FULL.html_icon}}
                    <p>openpilot can control the steering wheel down to 0 mph.</p>
                </div>
                <div class="rating-item">
                    {{Star.EMPTY.html_icon}}
                    <p>No steering control below certain speeds.</p>
                </div>
            </div>
            <div class="rating-row-item rating-row-item-secondary">
                <p>Steering Torque</p>
                <div class="rating-item">
                    {{Star.FULL.html_icon}}
                    <p>Car has enough steering torque for comfortable highway driving.</p>
                </div>
                <div class="rating-item">
                    {{Star.EMPTY.html_icon}}
                    <p>Limited ability to make turns.</p>
                </div>
            </div>
        </div>

        <div class="rating-row" style="border-bottom: none;">
            <div class="rating-row-item rating-row-item-primary">
                <img src="/supported-cars/icon-support.svg" alt="">
                <p>Support</p>
            </div>
            <div class="rating-row-item rating-row-item-secondary">
                <p>Actively Maintained</p>
                <div class="rating-item">
                    {{Star.FULL.html_icon}}
                    <p>Mainline software support, harness hardware sold by comma, lots of users, primary development target.</p>
                </div>
                <div class="rating-item">
                    {{Star.EMPTY.html_icon}}
                    <p>Low user count, community maintained, harness hardware not sold by comma.</p>
                </div>
            </div>
            <div class="rating-row-item rating-row-item-secondary"></div>
        </div>

    </div>
</section><!-- How we rate cars - section end -->

<section>
    <div class="container">
        <div class="table-container" role="table">
                    {% set footnote_tag = '<a style="position: absolute;" href="/vehicles/#footnote"><sup>{}</sup></a>' %}
                    {% for tier, car_rows in tiers %}

                    <div class="tier tier-{{tier.name.lower()}}">
                        <span class="font-Monument">{{tier.name.title()}} <span style="opacity:0.67;"><br>{{car_rows|length}} cars</span></span>
                        <p>The best openpilot experience. Great highway driving and beyond.</p>
                    </div>

                    <div class="flex-table header">
                        <div class="flex-row first col-1" role="columnheader">{{columns[0]}}</div>
                        <div class="flex-row col-2" role="columnheader">{{columns[1]}}</div>
                        <div class="flex-row col-3" role="columnheader">{{columns[2]}}</div>
                        <div class="flex-row mobile-hide col-even" role="columnheader">{{columns[3]}}</div>
                        <div class="flex-row mobile-hide col-even" role="columnheader">{{columns[4]}}</div>
                        <div class="flex-row mobile-hide col-even" role="columnheader">{{columns[5]}}</div>
                        <div class="flex-row mobile-hide col-even" role="columnheader">{{columns[6]}}</div>
                        <div class="flex-row mobile-hide col-even" role="columnheader">{{columns[7]}}</div>
                    </div>

                    {% for row in car_rows %}
                    <div class="flex-table row row-{{tier.name.lower()}}">
                        <div class="flex-row first col-1"><span class="make-icon make-{{row[0].text.lower()}}">{{row[0].text}}</span></div>
                        <div class="flex-row col-2">{{row[1].text}}<a href="#" style="display:none;" target="_blank" class="link-youtube"></a></div>
                        <div class="flex-row col-3">{{row[2].text}}</div>
                        {% for star_col in row if star_col.star is not none %}
                        <div class="flex-row flex-row-star col-even"><div class="flex-row-star-container">{{star_col.star.html_icon}}{{footnote_tag.format(star_col.footnote) if star_col.footnote else ''}}</div></div>
                        {% endfor %}
                    </div>
                    {% endfor %}

                    {% endfor %}
                </div>

        <div class="footnote-flex" id="footnote">
            <div>
                {% for footnote in footnotes[:4] %}
                <p class="table-footnote"><span id="">{{loop.index}}</span>{{footnote}}</p>
                {% endfor %}
            </div>
            <div>
                {% for footnote in footnotes[4:] %}
                <p class="table-footnote"><span id="">{{loop.index + 4}}</span>{{footnote}}</p>
                {% endfor %}
            </div>
        </div>
    </div>
</section>


<!-- FAQ ACCORDION -->

<section>
    <div class="container">

        <h1 class="title-FAQ">FREQUENTLY ASKED QUESTIONS</h1>
        <button class="accordion" @click="toggleFaq(0)">What is openpilot?</button>
            <div class="panel faq" v-bind:style="{maxHeight: getFaqHeight(0)}">
            <p>comma openpilot is an open source driver-assistance system. Currently, openpilot performs the functions of Adaptive Cruise Control (ACC) and Automated Lane Centering (ALC) for compatible vehicles. It performs similarly to Tesla Autopilot and GM Super Cruise. openpilot can steer, accelerate, and brake automatically for other vehicles within its lane. Check it out on <a href="https://github.com/commaai/openpilot" target="_self">GitHub</a>.</p>
            <p>In order to enforce driver alertness, openpilot includes a camera based Driver Monitoring (DM) system that alerts the driver when distracted or asleep. However, even with an attentive driver, we must make further efforts for the system to be safe. We have designed openpilot with two other safety considerations:</p>
            <ol>
                <li>The driver must always be capable to immediately retake manual control of the vehicle, by stepping on either pedal or by pressing the cancel button.</li>
                <li>The vehicle must not alter its trajectory too quickly for the driver to safely react. This means that while the system is engaged, the actuators are constrained to operate within reasonable limits.</li>
            </ol>
            </div>

            <button class="accordion" @click="toggleFaq(1)">How does openpilot work?</button>
            <div class="panel faq" v-bind:style="{maxHeight: getFaqHeight(1)}">
            <p>openpilot works by taking the radar data integrated with supported car models and combining it with the camera built into comma hardware, to determine what acceleration, braking, and steering events are required.</p>
            </div>

            <button class="accordion" @click="toggleFaq(2)">What should I buy to run openpilot in my car?</button>
            <div class="panel faq" v-bind:style="{maxHeight: getFaqHeight(2)}">
            <p>After verifying that your car is <a href="https://comma.ai/vehicles" target="_self">compatible</a>, we recommend purchasing a comma device in our <a href="https://comma.ai/shop" target="_self" rel="undefined">shop</a>. Don't forget to purchase a car harness to connect it to your vehicle. Add to cart, check out, and you’re good to go!</p>
            </div>

            <button class="accordion" @click="toggleFaq(3)">Do I have to pay attention?</button>
            <div class="panel faq" v-bind:style="{maxHeight: getFaqHeight(3)}">
            <p>Yes, the driver must always be able to immediately retake manual control of the vehicle, by stepping on either pedal or by pressing the cancel button. When openpilot is engaged, a driver monitoring system actively tracks driver awareness to help prevent distractions. The openpilot system disengages if you are distracted. Drivers must keep their eyes on the road at all times and be ready to take control of the car.</p>
            </div>

            <button class="accordion" @click="toggleFaq(4)">What are the limitations of openpilot Automated Lane Centering?</button>
            <div class="panel faq" v-bind:style="{maxHeight: getFaqHeight(4)}">
                <p>openpilot Automated Lane Centering (ALC) system&nbsp;does not automatically drive the vehicle or reduce the amount of attention that must be paid to the area in front of the vehicle. The driver must always keep control of the steering wheel and be ready to correct the ALC system action at all times.</p>
                <p><span style="font-weight: 400;">Many factors can impact the performance of openpilot ALC, causing it to be unable to function as intended. These include, but are not limited to:</span></p>
                <ul>
                <li><span style="font-weight: 400;">Poor visibility (heavy rain, snow, fog, etc.) or weather conditions are interfering with sensor operation.</span></li>
                <li>The road facing camera is obstructed, covered or damaged by<span style="font-weight: 400;">&nbsp;mud, ice, snow, etc.</span>
                </li>
                <li><span style="font-weight: 400;">Obstruction caused by applying excessive paint or adhesive products (such as wraps, stickers, rubber coating, etc.) onto the vehicle.</span></li>
                <li>The device is mounted incorrectly.</li>
                <li>When in sharp curves, like on-off ramps, intersections etc...; openpilot is designed to be limited in the amount of steering torque it can produce.</li>
                <li>In the presence of restricted lanes or construction zones.&nbsp;</li>
                <li>When driving on highly banked roads or in presence of strong cross-wind.</li>
                <li style="font-weight: 400;"><span style="font-weight: 400;">Extremely hot or cold temperatures.</span></li>
                <li style="font-weight: 400;"><span style="font-weight: 400;">Bright light (due to oncoming headlights, direct sunlight, etc.)</span></li>
                <li style="font-weight: 400;"><span style="font-weight: 400;">Driving on hills, narrow, or winding roads.</span></li>
                </ul>
                <p><span style="font-weight: 400;">The list above does not represent an exhaustive list of situations that may interfere with proper operation of openpilot components. It is the driver's responsibility to be in control of the vehicle at all times.</span></p>
            </div>

            <button class="accordion" @click="toggleFaq(5)">What are the limitations of openpilot Adaptive Cruise Control?</button>
            <div class="panel faq" v-bind:style="{maxHeight: getFaqHeight(5)}">
                <p>openpilot Adaptive Cruise Control (ACC) is not a system that allows careless or inattentive driving.&nbsp;It is still necessary for the driver to pay close attention to the vehicle’s surroundings and to be ready to re-take control of the gas and the brake at all times.</p>
                <p><span style="font-weight: 400;">Many factors can impact the performance of openpilot ACC, causing it to be unable to function as intended. These include, but are not limited to:</span></p>
                <ul>
                <li style="font-weight: 400;"><span style="font-weight: 400;">Poor visibility (heavy rain, snow, fog, etc.) or weather conditions are interfering with sensor operation.</span></li>
                <li style="font-weight: 400;">
                <span style="font-weight: 400;">The road facing camera or radar are obstructed, covered, or damaged&nbsp;</span><span style="font-weight: 400;">by mud, ice, snow, etc.</span>
                </li>
                <li style="font-weight: 400;"><span style="font-weight: 400;">Obstruction caused by applying excessive paint or adhesive products (such as wraps, stickers, rubber coating, etc.) onto the vehicle.</span></li>
                <li>The device is mounted incorrectly.</li>
                <li style="font-weight: 400;"><span style="font-weight: 400;">Approaching a toll booth.&nbsp;</span></li>
                <li>When driving on roads with pedestrians, cyclists, etc...</li>
                <li>In presence of traffic signs or stop lights, which are not detected by openpilot at this time.</li>
                <li>When the posted speed limit is below the user selected set speed. openpilot does not detect speed limits at this time.</li>
                <li>In presence of vehicles in the same lane that are not moving.</li>
                <li>When abrupt braking maneuvers are required. openpilot is designed to be limited in the amount of deceleration and acceleration that it can produce.</li>
                <li>When surrounding vehicles perform close cut-ins from neighbor lanes.</li>
                <li><span style="font-weight: 400;">Driving on hills, narrow, or winding roads.</span></li>
                <li><span style="font-weight: 400;">Extremely hot or cold temperatures.</span></li>
                <li><span style="font-weight: 400;">Bright light (due to oncoming headlights, direct sunlight, etc.)</span></li>
                <li><span style="font-weight: 400;">Interference from other equipment that generates ultrasonic waves.</span></li>
                </ul>
                <p><span style="font-weight: 400;">The list above does not represent an exhaustive list of situations that may interfere with proper operation of openpilot components. It is the driver's responsibility to be in control of the vehicle at all times.</span></p>
            </div>

            <button class="accordion" @click="toggleFaq(6)">Do I retain my car factory safety features with openpilot installed?</button>
            <div class="panel faq" v-bind:style="{maxHeight: getFaqHeight(6)}">
            <p>When openpilot is enabled in settings, Lane Keep Assist (LKAS), and Automated Lane Centering (ALC) are replaced by openpilot lateral control and only function when openpilot is engaged. Lane Departure Warning (LDW) works whether engaged or disengaged.<br><br>On certain cars, Adaptive Cruise Control (ACC) is replaced by openpilot longitudinal control.<br><br>openpilot preserves any other vehicle safety features, including, but are not limited to: AEB, auto high-beam, blind spot warning, and side collision warning.</p>
            </div>

            <button class="accordion" @click="toggleFaq(7)">Does openpilot support manual transmission cars (obviously, with the driver shifting)?</button>
            <div class="panel faq" v-bind:style="{maxHeight: getFaqHeight(7)}">
            <p>openpilot does not currently support manual transmission cars. If you’d like to learn more, join us on <a href="http://discord.comma.ai">Discord</a>, where some of our members are supporting manual cars.</p>
            </div>

            <button class="accordion" @click="toggleFaq(8)">How does openpilot recognize the car model it is connected to?</button>
            <div class="panel faq" v-bind:style="{maxHeight: getFaqHeight(8)}">
            <p>If your car is on the list of <a href="https://github.com/commaai/openpilot/blob/master/docs/CARS.md#supported-cars" target="_blank" rel="noopener">supported cars</a>, openpilot will automatically recognize the model of your car by performing a scan of relevant ECU firmware versions present in your car. The presence of certain ECU firmware versions is an indication of the model year, car brand, car model, and trim.</p>
            <p>If your car isn’t recognized as supported, your device will fall back to a dashcam only mode, preserving the stock functionalities and the user will receive a notification on the screen.</p>
            </div>

            <button class="accordion" @click="toggleFaq(9)">How do I update openpilot?</button>
            <div class="panel faq" v-bind:style="{maxHeight: getFaqHeight(9)}">
            <p>All software updates are received over-the-air (OTA). openpilot will automatically check for updates when connected to the internet. Your device will notify you on the screen when an update is available and prompt you to reboot the device to complete the update.</p>
            <p>You may also manually check for update in the settings, under the "software" tab.</p>
            </div>

            <button class="accordion" @click="toggleFaq(10)">Does openpilot work at all speeds?</button>
            <div class="panel faq" v-bind:style="{maxHeight: getFaqHeight(10)}">
            <p>Supported speeds vary depending on the car. Please reference the <a href="https://github.com/commaai/openpilot/blob/master/docs/CARS.md#supported-cars" target="_blank" rel="noopener">supported car list</a> for vehicle specific speed limitations. Maximum speed is the same as the maximum speed that stock ACC can be set to (car dependent) with a hard limit at ~84mph.</p>
            </div>

            <button class="accordion" @click="toggleFaq(11)">What is a fingerprint?</button>
            <div class="panel faq" v-bind:style="{maxHeight: getFaqHeight(11)}">
            <p>A fingerprint is the method openpilot uses to determine which vehicle it is connected to.</p>
            <p>Current methods use vehicle ECU firmware logging. openpilot will fingerprint the vehicle on each start up.</p>
            <p>If openpilot detects a firmware version not previously logged, it will need to be added to the codebase. Guides on adding new firmware can be found <a href="https://github.com/commaai/openpilot/wiki/Fingerprinting" target="_blank" rel="noopener">here</a>.</p>
            </div>

            <button class="accordion" @click="toggleFaq(12)">Where is my dongle ID?</button>
            <div class="panel faq" v-bind:style="{maxHeight: getFaqHeight(12)}">
            <p>The dongle ID of your device can be found in Settings of the device in the Device tab. </p>
            </div>

            <button class="accordion" @click="toggleFaq(13)">How do I leave feedback for openpilot?</button>
            <div class="panel faq" v-bind:style="{maxHeight: getFaqHeight(13)}">
            <p>Feedback, good or bad, can be given in the #openpilot-experience channel on our community <a href="https://discord.comma.ai" target="_blank" rel="noopener">Discord</a>.</p>
            </div>
    </div>
</section>



<section class="cta-secondary" style="background: #EBEEF5;">
    <div class="container">
        <div class="two-column-lauout">
            <div class="margin-bottom-large">
                <img src="/supported-cars/icon-discord.svg" style="width: 48px;" alt="Icon Frequently Asked Questions">
                <h2>Join the conversation</h2>
                <p class="font-large">Have a question or want to learn more?  There are thousands of knowledgeable community members on the Discord; most car makes have a dedicated channel!</p>
                <a href="https://discord.com/invite/avCJxEX" class="btn-regular" target="_blank">Join our Discord<span class="icon"></span></a>
            </div>
            <div>
                <img src="/supported-cars/icon-github.svg" style="width: 48px;" alt="Icon GitHub">
                <h2>Developer Resources</h2>
                <p class="font-large">Check out the code behind openpilot and learn how to add support for your own car. Review, fork, and contribute to the open source ecosystem.</p>
                <a href="https://github.com/commaai/openpilot" class="btn-regular" target="_blank">Visit our GitHub<span class="icon"></span></a>
            </div>


        </div>
    </div>
</section>


    <section class="section-buy">
        <div class="container">
            <div class="section-buy-flex">
                <h2>Try the comma three with our 30-day money back trial.</h2>
                <a href="https://comma.ai/shop/products/three" class="btn-regular btn-regular-green">Buy Now</a>
            </div>

        </div>
    </section>
</div>

</template>


<script>
export default {
  data() {
    return {
      expandedFaqs: [],
      faqHeights: []
    }
  },
  methods: {
    toggleFaq(idx) {
      if (this.expandedFaqs.includes(idx)) {
        this.expandedFaqs = this.expandedFaqs.filter(q => q !== idx)
      } else {
        this.expandedFaqs.push(idx)
      }
    },
    getFaqHeight(idx) {
      return this.expandedFaqs.includes(idx) ? this.faqHeights[idx] + 'px' : 0
    },
    getFaqHeights() {
      this.faqHeights = Array.from(document.querySelectorAll('.panel.faq')).map(x => x.scrollHeight)
    }
  },
  mounted () {
    this.getFaqHeights()
    window.addEventListener('resize', this.getFaqHeights)
  },
}
</script>
