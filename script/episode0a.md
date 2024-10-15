# Introduction
Ever since I was tall enough to ride, I've
been impressed by roller coasters. Even before 
that, I was interested in space. So when I
first went on Space Mountain in Disneyland,
CA, I was impressed and changed.

One of the things that impressed me the most
wasn't actually even the ride itself, it was
the mystery. I like being able to see the
hidden aspects of the world. Being a dark
indoor roller coaster, it was hidden in ways
that most coasters aren't.

One of the things that reveals it was the model
was a model. I think it was the one that the
engineers built while planning and constructing
the real thing.  I saw it while it was on
exhibit in a pop-up art museum in New Orleans
Square.

Having seen the twists and turns compacted into
such a small space, I wasn't satisfied. I wanted
to keep it. I wanted to be able to take the whole
experience with me. Since that's obviously
impossible, I wanted a model of it.

Now the plans to Space Mountain aren't available
to me (as far as I know) and even so, I like
to do things on my own. So I came up with the
idea of carrying an inertial measurement system
on the roller coaster. I would take the measurements,
then easily integrate the measurements and plot
the data as the course of the roller coaster.
In the beginning, it seemed like taking the
measurement was the hard part.

At just about the time I had learned enough
about electronics to contemplate building my own
systems, I started hearing about MEMS sensors.
In the beginning they were expensive, inaccurate,
and analog. I originall considered using the
voltage output to drive a voltage-to-frequency
modulator and record the result on audio tape.

Several years later, I graduated college and
regained interest in this topic. In the intervening
time, MEMS hardware advanced considerably. Instead
of one axis in a chip, there were now two and three axis
sensors. Some even spoke digital. At the same time,
I learned about microcontrollers and designed a system
to use the small microcontroller on the Sparkfun Logomatic
to record the output of these sensors. I built
them into a plastic box which just barely fit in
my pocket, and in April 2009, recorded my first
rollercoasterometer data. I no longer have that
device, but I still have the data.

As the years progressed, more and more axes
were integrated into one chip. Soon you could
get six-axis sensors with all three 
accelerometer axes and also three rotation axes.
Magnetometers became available that could sense
the Earth's magnetic field in three dimensions,
to act as a compass. I kept getting new sensors,
and kept recording more data.

Then the smart-phone came along. Nowadays, most
people carry a full set of IMU sensors (often
including magnetic and pressure sensors) in our
pockets on a daily basis. I wrote an app to record that
data, along with the GPS position that the phone
also supplies.

Nowadays then, recording data is easy -- just turn
on the app, slip it in your pocket, and record the data.
I've done this dozens of times, but I still don't
have a system to analyze this data. I know it's possible -- 
they did it in the sixties on Apollo. I just
haven't been able to do it myself yet. But I think
I'm close, and I want to document my finishing steps.

In this series, we will explore the various
branches of mathematics, physics, and technology
that are necessary to solve the problem: What is
the course of the Space Mountain track?
