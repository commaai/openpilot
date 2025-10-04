# Copyright (c) 2014 Sandstorm Development Group, Inc. and contributors
# Licensed under the MIT License:
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

@0xb8630836983feed7;

$import "/capnp/c++.capnp".namespace("capnp");

interface Persistent@0xc8cb212fcd9f5691(SturdyRef, Owner) {
  # Interface implemented by capabilities that outlive a single connection. A client may save()
  # the capability, producing a SturdyRef. The SturdyRef can be stored to disk, then later used to
  # obtain a new reference to the capability on a future connection.
  #
  # The exact format of SturdyRef depends on the "realm" in which the SturdyRef appears. A "realm"
  # is an abstract space in which all SturdyRefs have the same format and refer to the same set of
  # resources. Every vat is in exactly one realm. All capability clients within that vat must
  # produce SturdyRefs of the format appropriate for the realm.
  #
  # Similarly, every VatNetwork also resides in a particular realm. Usually, a vat's "realm"
  # corresponds to the realm of its main VatNetwork. However, a Vat can in fact communicate over
  # a VatNetwork in a different realm -- in this case, all SturdyRefs need to be transformed when
  # coming or going through said VatNetwork. The RPC system has hooks for registering
  # transformation callbacks for this purpose.
  #
  # Since the format of SturdyRef is realm-dependent, it is not defined here. An application should
  # choose an appropriate realm for itself as part of its design. Note that under Sandstorm, every
  # application exists in its own realm and is therefore free to define its own SturdyRef format;
  # the Sandstorm platform handles translating between realms.
  #
  # Note that whether a capability is persistent is often orthogonal to its type. In these cases,
  # the capability's interface should NOT inherit `Persistent`; instead, just perform a cast at
  # runtime. It's not type-safe, but trying to be type-safe in these cases will likely lead to
  # tears. In cases where a particular interface only makes sense on persistent capabilities, it
  # still should not explicitly inherit Persistent because the `SturdyRef` and `Owner` types will
  # vary between realms (they may even be different at the call site than they are on the
  # implementation). Instead, mark persistent interfaces with the $persistent annotation (defined
  # below).
  #
  # Sealing
  # -------
  #
  # As an added security measure, SturdyRefs may be "sealed" to a particular owner, such that
  # if the SturdyRef itself leaks to a third party, that party cannot actually restore it because
  # they are not the owner. To restore a sealed capability, you must first prove to its host that
  # you are the rightful owner. The precise mechanism for this authentication is defined by the
  # realm.
  #
  # Sealing is a defense-in-depth mechanism meant to mitigate damage in the case of catastrophic
  # attacks. For example, say an attacker temporarily gains read access to a database full of
  # SturdyRefs: it would be unfortunate if it were then necessary to revoke every single reference
  # in the database to prevent the attacker from using them.
  #
  # In general, an "owner" is a course-grained identity. Because capability-based security is still
  # the primary mechanism of security, it is not necessary nor desirable to have a separate "owner"
  # identity for every single process or object; that is exactly what capabilities are supposed to
  # avoid! Instead, it makes sense for an "owner" to literally identify the owner of the machines
  # where the capability is stored. If untrusted third parties are able to run arbitrary code on
  # said machines, then the sandbox for that code should be designed using Distributed Confinement
  # such that the third-party code never sees the bits of the SturdyRefs and cannot directly
  # exercise the owner's power to restore refs. See:
  #
  #     http://www.erights.org/elib/capability/dist-confine.html
  #
  # Resist the urge to represent an Owner as a simple public key. The whole point of sealing is to
  # defend against leaked-storage attacks. Such attacks can easily result in the owner's private
  # key being stolen as well. A better solution is for `Owner` to contain a simple globally unique
  # identifier for the owner, and for everyone to separately maintain a mapping of owner IDs to
  # public keys. If an owner's private key is compromised, then humans will need to communicate
  # and agree on a replacement public key, then update the mapping.
  #
  # As a concrete example, an `Owner` could simply contain a domain name, and restoring a SturdyRef
  # would require signing a request using the domain's private key. Authenticating this key could
  # be accomplished through certificate authorities or web-of-trust techniques.

  save @0 SaveParams -> SaveResults;
  # Save a capability persistently so that it can be restored by a future connection.  Not all
  # capabilities can be saved -- application interfaces should define which capabilities support
  # this and which do not.

  struct SaveParams {
    sealFor @0 :Owner;
    # Seal the SturdyRef so that it can only be restored by the specified Owner. This is meant
    # to mitigate damage when a SturdyRef is leaked. See comments above.
    #
    # Leaving this value null may or may not be allowed; it is up to the realm to decide. If a
    # realm does allow a null owner, this should indicate that anyone is allowed to restore the
    # ref.
  }
  struct SaveResults {
    sturdyRef @0 :SturdyRef;
  }
}

annotation persistent(interface, field) :Void;
# Apply this annotation to interfaces for objects that will always be persistent, instead of
# extending the Persistent capability, since the correct type parameters to Persistent depend on
# the realm, which is orthogonal to the interface type and therefore should not be defined
# along-side it.
#
# You may also apply this annotation to a capability-typed field which will always contain a
# persistent capability, but where the capability's interface itself is not already marked
# persistent.
#
# Note that absence of the $persistent annotation doesn't mean a capability of that type isn't
# persistent; it just means not *all* such capabilities are persistent.
