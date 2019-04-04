import unittest

from websocket._cookiejar import SimpleCookieJar

try:
    import Cookie
except:
    import http.cookies as Cookie


class CookieJarTest(unittest.TestCase):
    def testAdd(self):
        cookie_jar = SimpleCookieJar()
        cookie_jar.add("")
        self.assertFalse(cookie_jar.jar, "Cookie with no domain should not be added to the jar")

        cookie_jar = SimpleCookieJar()
        cookie_jar.add("a=b")
        self.assertFalse(cookie_jar.jar, "Cookie with no domain should not be added to the jar")

        cookie_jar = SimpleCookieJar()
        cookie_jar.add("a=b; domain=.abc")
        self.assertTrue(".abc" in cookie_jar.jar)

        cookie_jar = SimpleCookieJar()
        cookie_jar.add("a=b; domain=abc")
        self.assertTrue(".abc" in cookie_jar.jar)
        self.assertTrue("abc" not in cookie_jar.jar)

        cookie_jar = SimpleCookieJar()
        cookie_jar.add("a=b; c=d; domain=abc")
        self.assertEquals(cookie_jar.get("abc"), "a=b; c=d")

        cookie_jar = SimpleCookieJar()
        cookie_jar.add("a=b; c=d; domain=abc")
        cookie_jar.add("e=f; domain=abc")
        self.assertEquals(cookie_jar.get("abc"), "a=b; c=d; e=f")

        cookie_jar = SimpleCookieJar()
        cookie_jar.add("a=b; c=d; domain=abc")
        cookie_jar.add("e=f; domain=.abc")
        self.assertEquals(cookie_jar.get("abc"), "a=b; c=d; e=f")

        cookie_jar = SimpleCookieJar()
        cookie_jar.add("a=b; c=d; domain=abc")
        cookie_jar.add("e=f; domain=xyz")
        self.assertEquals(cookie_jar.get("abc"), "a=b; c=d")
        self.assertEquals(cookie_jar.get("xyz"), "e=f")
        self.assertEquals(cookie_jar.get("something"), "")

    def testSet(self):
        cookie_jar = SimpleCookieJar()
        cookie_jar.set("a=b")
        self.assertFalse(cookie_jar.jar, "Cookie with no domain should not be added to the jar")

        cookie_jar = SimpleCookieJar()
        cookie_jar.set("a=b; domain=.abc")
        self.assertTrue(".abc" in cookie_jar.jar)

        cookie_jar = SimpleCookieJar()
        cookie_jar.set("a=b; domain=abc")
        self.assertTrue(".abc" in cookie_jar.jar)
        self.assertTrue("abc" not in cookie_jar.jar)

        cookie_jar = SimpleCookieJar()
        cookie_jar.set("a=b; c=d; domain=abc")
        self.assertEquals(cookie_jar.get("abc"), "a=b; c=d")

        cookie_jar = SimpleCookieJar()
        cookie_jar.set("a=b; c=d; domain=abc")
        cookie_jar.set("e=f; domain=abc")
        self.assertEquals(cookie_jar.get("abc"), "e=f")

        cookie_jar = SimpleCookieJar()
        cookie_jar.set("a=b; c=d; domain=abc")
        cookie_jar.set("e=f; domain=.abc")
        self.assertEquals(cookie_jar.get("abc"), "e=f")

        cookie_jar = SimpleCookieJar()
        cookie_jar.set("a=b; c=d; domain=abc")
        cookie_jar.set("e=f; domain=xyz")
        self.assertEquals(cookie_jar.get("abc"), "a=b; c=d")
        self.assertEquals(cookie_jar.get("xyz"), "e=f")
        self.assertEquals(cookie_jar.get("something"), "")

    def testGet(self):
        cookie_jar = SimpleCookieJar()
        cookie_jar.set("a=b; c=d; domain=abc.com")
        self.assertEquals(cookie_jar.get("abc.com"), "a=b; c=d")
        self.assertEquals(cookie_jar.get("x.abc.com"), "a=b; c=d")
        self.assertEquals(cookie_jar.get("abc.com.es"), "")
        self.assertEquals(cookie_jar.get("xabc.com"), "")

        cookie_jar.set("a=b; c=d; domain=.abc.com")
        self.assertEquals(cookie_jar.get("abc.com"), "a=b; c=d")
        self.assertEquals(cookie_jar.get("x.abc.com"), "a=b; c=d")
        self.assertEquals(cookie_jar.get("abc.com.es"), "")
        self.assertEquals(cookie_jar.get("xabc.com"), "")
