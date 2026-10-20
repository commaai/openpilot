// capnp schema compiler: turns cereal's .capnp sources into schema.json's text, byte for byte what gen_schema.py
// writes from pycapnp, so a log can be read with the exact schema it was written with. The parser covers the
// language cereal and its forks use (structs, enums, unions, groups, generics, consts, usings, imports,
// annotations); the layout is a port of capnp's StructLayout and StructTranslator (compiler/node-translator.c++).

const PRIMITIVE = {Bool: 0, Int8: 3, Int16: 4, Int32: 5, Int64: 6, UInt8: 3, UInt16: 4, UInt32: 5, UInt64: 6, Float32: 5, Float64: 6};
const CODE = {Bool: "b", Int8: "i8", Int16: "i16", Int32: "i32", Int64: "i64", UInt8: "u8", UInt16: "u16", UInt32: "u32",
  UInt64: "u64", Float32: "f32", Float64: "f64", Text: "t", Data: "d", Void: "v", AnyPointer: "p", AnyStruct: "p",
  AnyList: "p", Capability: "p"};
const BUILTIN = new Set([...Object.keys(CODE), "List"]);

// ---- tokens and statements

const TOKEN = /\s+|#[^\n]*|(0x[0-9a-fA-F]+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)|([A-Za-z_]\w*)|("(?:[^"\\]|\\.)*")|([@:=()[\]{},;.$!+\-*/])/y;

function tokenize(text) {
  const out = [];
  TOKEN.lastIndex = 0;
  while (TOKEN.lastIndex < text.length) {
    const at = TOKEN.lastIndex, m = TOKEN.exec(text);
    if (!m) throw new Error(`capnp: unexpected ${JSON.stringify(text.slice(at, at + 20))}`);
    if (m[1]) out.push({num: m[1]});
    else if (m[2]) out.push({id: m[2]});
    else if (m[3]) out.push({str: JSON.parse(m[3])});
    else if (m[4]) out.push({op: m[4]});
  }
  return out;
}

const isOp = (t, op) => t?.op === op;

// a block is a list of statements, each the tokens before its ";" or "{", plus the nested block of a "{"
function statements(toks, k = 0, nested = false) {
  const out = [];
  let cur = [];
  for (; k < toks.length; k++) {
    const t = toks[k];
    if (isOp(t, ";")) {
      out.push({toks: cur, block: null});
      cur = [];
    } else if (isOp(t, "{")) {
      const [block, end] = statements(toks, k + 1, true);
      out.push({toks: cur, block});
      cur = [];
      k = end;
    } else if (isOp(t, "}")) {
      if (!nested) throw new Error("capnp: unbalanced }");
      return [out, k];
    } else {
      cur.push(t);
    }
  }
  if (nested || cur.length) throw new Error("capnp: unexpected end of file");
  return [out, k];
}

// ---- expressions: names, members, applications, imports and literals

class Cursor {
  constructor(toks) { this.toks = toks; this.k = 0; }
  peek(d = 0) { return this.toks[this.k + d]; }
  next() { return this.toks[this.k++]; }
  op(o) { if (isOp(this.peek(), o)) { this.k++; return true; } return false; }
  expect(o) { if (!this.op(o)) throw new Error(`capnp: expected ${o} near ${JSON.stringify(this.peek())}`); }
  id() {
    const t = this.next();
    if (!t?.id) throw new Error(`capnp: expected a name near ${JSON.stringify(t)}`);
    return t.id;
  }
  get done() { return this.k >= this.toks.length; }

  expr() {
    if (this.op("-")) return {neg: this.expr()};
    let e;
    const t = this.next();
    if (!t) throw new Error("capnp: expected an expression");
    if (t.num) e = {num: t.num};
    else if (t.str !== undefined) e = {str: t.str};
    else if (t.id === "import") e = {import: this.next().str};
    else if (t.id) e = {name: t.id};
    else if (isOp(t, ".")) e = {abs: this.id()};
    else if (isOp(t, "(")) e = {tuple: this.list(")")};
    else if (isOp(t, "[")) e = {list: this.list("]")};
    else throw new Error(`capnp: unexpected ${JSON.stringify(t)}`);
    for (;;) {
      if (this.op(".")) e = {of: e, member: this.id()};
      else if (isOp(this.peek(), "(") && (e.name || e.of || e.abs)) { this.next(); e = {app: e, args: this.list(")")}; }
      else return e;
    }
  }

  // comma-separated expressions (or name = value pairs) up to the closing bracket
  list(close) {
    const out = [];
    while (!this.op(close)) {
      if (this.peek()?.id && isOp(this.peek(1), "=")) { this.k += 2; out.push(this.expr()); }
      else out.push(this.expr());
      this.op(",");
    }
    return out;
  }

  // annotations are skipped: $name, $a.b, $name(args)
  annotations() {
    while (this.op("$")) {
      this.expr();
    }
  }
}

// ---- declarations and scopes

// parses one file into a scope tree: {kind, name, members: Map, parent, ...}
function parseFile(text, path) {
  const file = {kind: "file", name: "", path, members: new Map(), parent: null};
  declare(file, statements(tokenize(text))[0], file);
  return file;
}

function declare(scope, stmts, file) {
  for (const s of stmts) {
    const c = new Cursor(s.toks);
    const head = c.peek()?.id;
    if (isOp(c.peek(), "@") || isOp(c.peek(), "$")) continue;  // file id, file annotation
    if (head === "using") {
      c.next();
      const name = isOp(c.peek(1), "=") ? c.id() : null;
      if (name) c.next();
      const expr = c.expr();
      const alias = name || (expr.member ?? expr.name);
      scope.members.set(alias, {kind: "using", name: alias, expr, scope});
    } else if (head === "struct" || head === "enum" || head === "interface") {
      c.next();
      const name = c.id();
      if (c.op("@")) c.next();
      const params = [];
      if (c.op("(")) while (!c.op(")")) { params.push(c.id()); c.op(","); }
      const display = scope.kind === "file" ? name : `${scope.display}.${name}`;
      const node = {kind: head, name, display, params, parent: scope, file, members: new Map(), body: s.block || []};
      scope.members.set(name, node);
      if (head === "struct") declare(node, node.body.filter((m) => isNested(m)), file);
      if (head === "enum") node.enumerants = enumerants(node.body);
    } else if (head === "const") {
      c.next();
      const name = c.id();
      if (c.op("@")) c.next();
      c.expect(":");
      const type = c.expr();
      c.expect("=");
      scope.members.set(name, {kind: "const", name, type, value: c.expr(), scope});
    }
    // annotation declarations and anything unrecognized at this level carry no layout
  }
}

const isNested = (s) => ["struct", "enum", "interface", "using", "const", "annotation"].includes(s.toks[0]?.id) && !isOp(s.toks[1], "@") && !isOp(s.toks[1], ":");

// enumerant names by ordinal, which is also their value
function enumerants(body) {
  const out = [];
  for (const s of body) {
    const c = new Cursor(s.toks);
    if (!c.peek()?.id || !isOp(c.peek(1), "@")) continue;
    const name = c.id();
    c.next();
    out[Number(c.next().num)] = name;
  }
  return out;
}

class Compiler {
  // files: {path: text}; resolve(from, import) gives the path an import names, or null for files that only carry
  // annotations (c++.capnp, java.capnp)
  constructor(files, resolve) {
    this.files = files;
    this.resolve = resolve;
    this.parsed = new Map();
    this.layouts = new Map();
  }

  file(path) {
    let f = this.parsed.get(path);
    if (!f) {
      if (!(path in this.files)) throw new Error(`capnp: missing ${path}`);
      this.parsed.set(path, f = parseFile(this.files[path], path));
    }
    return f;
  }

  // the declaration a name refers to from scope: nested members and generic parameters outward, then builtins
  lookup(scope, name) {
    for (let s = scope; s; s = s.parent) {
      if (s.params?.includes(name)) return {kind: "param"};
      const m = s.members.get(name);
      if (m) return this.deref(m);
    }
    if (BUILTIN.has(name)) return {kind: "builtin", name};
    if (name === "true" || name === "false" || name === "inf" || name === "nan") return {kind: "builtin", name};
    throw new Error(`capnp: unknown name ${name}`);
  }

  deref(m) {
    if (m.kind !== "using") return m;
    m.target ||= this.entity(m.expr, m.scope);
    return m.target;
  }

  entity(e, scope) {
    if (e.import !== undefined) {
      const path = this.resolve(scope.file?.path ?? scope.path, e.import);
      return path === null ? {kind: "file", members: new Map(), parent: null} : this.file(path);
    }
    if (e.name) return this.lookup(scope, e.name);
    if (e.abs) {
      let f = scope;
      while (f.parent) f = f.parent;
      return this.deref(f.members.get(e.abs) ?? (() => { throw new Error(`capnp: unknown name .${e.abs}`); })());
    }
    if (e.of) {
      const of = this.entity(e.of, scope), m = of.members?.get(e.member);
      if (!m) throw new Error(`capnp: unknown member ${e.member}`);
      return this.deref(m);
    }
    if (e.app) return this.entity(e.app, scope);
    throw new Error("capnp: not a name");
  }

  // a field's type: {code, lg (layout size: -1 void, -2 pointer, else log2 bits), node?, elem?}
  type(e, scope) {
    const d = this.entity(e, scope);
    if (d.kind === "builtin") {
      if (d.name === "List") return {code: "l", lg: -2, elem: this.type(e.args[0], scope)};
      if (d.name in PRIMITIVE) return {code: CODE[d.name], lg: PRIMITIVE[d.name], prim: d.name};
      if (d.name === "Void") return {code: "v", lg: -1};
      if (CODE[d.name]) return {code: CODE[d.name], lg: -2};
    }
    if (d.kind === "param" || d.kind === "interface") return {code: "p", lg: -2};
    if (d.kind === "struct") return {code: "s", lg: -2, node: d};
    if (d.kind === "enum") return {code: "e", lg: 4, node: d};
    throw new Error(`capnp: not a type: ${JSON.stringify(e)}`);
  }

  // ---- struct layout: the members of a struct (and of its groups) in StructTranslator's order

  layout(struct) {
    let out = this.layouts.get(struct);
    if (out) return out;
    const top = new Top();
    const root = new Member(null, struct.display, false);
    const byOrdinal = [], all = [];
    const add = (m, ordinal) => { all.push(m); if (ordinal !== undefined) byOrdinal.push([ordinal, byOrdinal.length, m]); };

    const field = (c, parent, scope, inUnion) => {
      const name = c.id();
      c.expect("@");
      const ordinal = Number(c.next().num);
      c.expect(":");
      const m = new Member(parent, name, inUnion);
      m.typeExpr = c.expr();
      if (c.op("=")) m.defaultExpr = c.expr();
      m.scope = scope;
      parent.childCount++;
      add(m, ordinal);
    };

    // a named union or group becomes its own node, named after its parent
    const group = (parent, name, inUnion) => {
      parent.childCount++;
      const m = new Member(parent, `${parent.display}.${name}`, inUnion, name);
      add(m);
      return m;
    };

    const unionBody = (stmts, parent, union) => {
      for (const s of stmts) {
        const c = new Cursor(s.toks), kind = memberKind(s);
        if (kind === "field") field(c, parent, new Group(union), true);
        else if (kind === "union") {
          const {name, ordinal} = unionHead(c);
          const m = group(parent, name, true);
          m.union = new Union(new Group(union));
          unionBody(s.block, m, m.union);
          if (ordinal !== undefined) byOrdinal.push([ordinal, byOrdinal.length, m]);
        } else if (kind === "group") {
          const m = group(parent, c.id(), true);
          body(s.block, m, new Group(union));
        }
      }
    };

    const body = (stmts, parent, scope) => {
      for (const s of stmts) {
        const c = new Cursor(s.toks), kind = memberKind(s);
        if (kind === "field") field(c, parent, scope, false);
        else if (kind === "union") {
          const {name, ordinal} = unionHead(c);
          const union = new Union(scope);
          const m = name ? group(parent, name, false) : parent;
          m.union = union;
          unionBody(s.block, m, union);
          if (ordinal !== undefined) byOrdinal.push([ordinal, byOrdinal.length, m]);
        } else if (kind === "group") {
          body(s.block, group(parent, c.id(), false), scope);
        }
      }
    };
    body(struct.body, root, top);

    // offsets are assigned in ordinal order; ties keep declaration order, like the multimap
    byOrdinal.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    for (const [, , m] of byOrdinal) {
      m.schema();
      if (!m.typeExpr) {
        m.union.addDiscriminant();
        continue;
      }
      m.decl = struct;
      m.type = this.type(m.typeExpr, struct);
      const lg = m.type.lg;
      m.offset = lg === -2 ? m.scope.addPointer() : lg === -1 ? (m.scope.addVoid(), 0) : m.scope.addData(lg);
    }
    root.finishGroup();
    for (const m of all) {
      if (!m.typeExpr) m.finishGroup();
      m.schema();
    }
    for (const m of [root, ...all.filter((m) => !m.typeExpr)]) {
      m.data = top.dataWordCount;
      m.ptrs = top.pointerCount;
    }
    out = root;
    this.layouts.set(struct, out);
    return out;
  }

  // ---- defaults: only scalar and enum defaults are written (as gen_schema.py does)

  value(e, type, scope) {
    if (e.neg) {
      return -this.value(e.neg, type, scope);
    }
    if (e.num) {
      const n = e.num;
      if (type.prim.startsWith("Float")) return Number(/^0x/.test(n) ? BigInt(n) : n);
      if (/[.eE]/.test(n) && !/^0x/.test(n)) throw new Error(`capnp: ${n} is not an integer`);
      return /^0\d/.test(n) ? BigInt(`0o${n.slice(1)}`) : BigInt(n);
    }
    if (e.name === "true" || e.name === "false") return e.name === "true";
    if (e.name === "inf") return Infinity;
    if (e.name === "nan") return NaN;
    const d = this.entity(e, scope);
    if (d.kind === "const") return this.value(d.value, type, d.scope);
    throw new Error(`capnp: unsupported default ${JSON.stringify(e)}`);
  }

  // the default's raw bits as gen_schema.py's bits() gives them, or undefined when it's zero
  bits(m) {
    const t = m.type;
    if (t.code === "e") {
      const k = t.node.enumerants.indexOf(m.defaultExpr.name);
      if (k < 0) throw new Error(`capnp: unknown enumerant ${m.defaultExpr.name}`);
      return k || undefined;
    }
    if (!t.prim) return undefined;
    const v = this.value(m.defaultExpr, t, m.decl);
    if (t.prim === "Bool") return v ? 1 : undefined;
    if (t.prim === "Float32") {
      const f = new Float32Array([v]);
      return f[0] !== 0 ? new Uint32Array(f.buffer)[0] : undefined;
    }
    if (t.prim === "Float64") {
      if (v === 0) return undefined;
      const view = new DataView(new ArrayBuffer(8));
      view.setFloat64(0, v, true);
      return view.getBigUint64(0, true);
    }
    const n = BigInt(v);
    return n !== 0n ? n : undefined;
  }

  // ---- output: schema.json's {structs, enums}, structs in first-reference order from the root

  emit(rootStruct) {
    const structs = [], enums = [], index = new Map(), enumIndex = new Map();
    const addEnum = (node) => {
      if (!enumIndex.has(node)) {
        enumIndex.set(node, enums.length);
        enums.push(node.enumerants);
      }
      return enumIndex.get(node);
    };
    const desc = (t) => t.code === "s" ? ["s", addStruct(this.layout(t.node))] : t.code === "e" ? ["e", addEnum(t.node)] : t.code === "l" ? ["l", desc(t.elem)] : [t.code];
    const addStruct = (node) => {
      if (index.has(node)) return index.get(node);
      const entry = {name: node.display, data: node.data, ptrs: node.ptrs, disc: node.discCount ? node.discOffset : -1, fields: []};
      index.set(node, structs.length);
      structs.push(entry);
      for (const f of node.fields) {
        const disc = f.disc ?? -1;
        if (!f.typeExpr) {
          entry.fields.push([f.name, disc, "g", 0, addStruct(f)]);
          continue;
        }
        const d = desc(f.type);
        const field = [f.name, disc, d[0], f.offset, d.length > 1 ? d[1] : null];
        if (f.defaultExpr && (f.type.prim || f.type.code === "e")) {
          const b = this.bits(f);
          if (b !== undefined) field.push(b);
        }
        entry.fields.push(field);
      }
      return index.get(node);
    };
    addStruct(this.layout(rootStruct));
    return json({structs, enums});
  }
}

function memberKind(s) {
  const t = s.toks;
  if (t[0]?.id === "union" && t.length === 1 || t.some((x) => x.id === "union") && s.block && (isOp(t[1], "@") || isOp(t[1], ":"))) return "union";
  if (s.block && isOp(t[1], ":") && t[2]?.id === "group") return "group";
  if (!s.block && t[0]?.id && isOp(t[1], "@")) return "field";
  return null;
}

function unionHead(c) {
  if (c.peek()?.id === "union") return {name: ""};
  const name = c.id();
  let ordinal;
  if (c.op("@")) {
    ordinal = Number(c.next().num);
    c.op("!");
  }
  return {name, ordinal};
}

// a member of a struct: a field, a group or a named union (both of which are also nodes with fields of their own)
class Member {
  constructor(parent, name, isInUnion, short = name) {
    this.parent = parent;
    this.display = name;
    this.name = short;
    this.isInUnion = isInUnion;
    this.childCount = 0;
    this.discCount = 0;
    this.fields = [];
    this.inited = false;
  }

  // a member's index (and discriminant) is decided when its schema is first needed, in ordinal order
  schema() {
    if (this.inited || !this.parent) return;
    this.inited = true;
    const p = this.parent;
    if (!p.fields.length) p.schema();
    p.fields.push(this);
    if (this.isInUnion) this.disc = p.discCount++;
  }

  finishGroup() {
    if (this.union) {
      this.union.addDiscriminant();
      this.discOffset = this.union.discriminantOffset;
    }
  }
}

// ---- StructLayout, ported line for line

const HOLES = 6;

class HoleSet {
  constructor() { this.holes = [0, 0, 0, 0, 0, 0]; }

  tryAllocate(lg) {
    if (lg >= HOLES) return null;
    if (this.holes[lg] !== 0) {
      const r = this.holes[lg];
      this.holes[lg] = 0;
      return r;
    }
    const next = this.tryAllocate(lg + 1);
    if (next === null) return null;
    const r = next * 2;
    this.holes[lg] = r + 1;
    return r;
  }

  addHolesAtEnd(lg, offset, limit = HOLES) {
    while (lg < limit) {
      this.holes[lg] = offset;
      lg++;
      offset = (offset + 1) >>> 1;
    }
  }

  tryExpand(oldLg, oldOffset, factor) {
    if (factor === 0) return true;
    if (oldLg === HOLES) return false;
    if (this.holes[oldLg] !== oldOffset + 1) return false;
    if (this.tryExpand(oldLg + 1, oldOffset >>> 1, factor - 1)) {
      this.holes[oldLg] = 0;
      return true;
    }
    return false;
  }

  smallestAtLeast(size) {
    for (let i = size; i < HOLES; i++) if (this.holes[i] !== 0) return i;
    return null;
  }
}

class Top {
  constructor() {
    this.dataWordCount = 0;
    this.pointerCount = 0;
    this.holes = new HoleSet();
  }

  addVoid() {}

  addData(lg) {
    const hole = this.holes.tryAllocate(lg);
    if (hole !== null) return hole;
    const offset = this.dataWordCount++ * 2 ** (6 - lg);
    this.holes.addHolesAtEnd(lg, offset + 1);
    return offset;
  }

  addPointer() { return this.pointerCount++; }

  tryExpandData(oldLg, oldOffset, factor) { return this.holes.tryExpand(oldLg, oldOffset, factor); }
}

class Union {
  constructor(parent) {
    this.parent = parent;
    this.groupCount = 0;
    this.discriminantOffset = null;
    this.dataLocations = [];
    this.pointerLocations = [];
  }

  addNewDataLocation(lg) {
    const offset = this.parent.addData(lg);
    this.dataLocations.push({lg, offset});
    return offset;
  }

  addNewPointerLocation() {
    const p = this.parent.addPointer();
    this.pointerLocations.push(p);
    return p;
  }

  newGroupAddingFirstMember() {
    if (++this.groupCount === 2) this.addDiscriminant();
  }

  addDiscriminant() {
    if (this.discriminantOffset !== null) return false;
    this.discriminantOffset = this.parent.addData(4);
    return true;
  }

  tryExpandTo(loc, newLg) {
    if (newLg <= loc.lg) return true;
    if (!this.parent.tryExpandData(loc.lg, loc.offset, newLg - loc.lg)) return false;
    loc.offset >>>= newLg - loc.lg;
    loc.lg = newLg;
    return true;
  }
}

// how much of one of the union's data locations a group uses
class Usage {
  constructor(lg = null) {
    this.isUsed = lg !== null;
    this.lgSizeUsed = lg ?? 0;
    this.holes = new HoleSet();
  }

  smallestHoleAtLeast(loc, lg) {
    if (!this.isUsed) return lg <= loc.lg ? loc.lg : null;
    if (lg >= this.lgSizeUsed) return lg < loc.lg ? lg : null;
    const hole = this.holes.smallestAtLeast(lg);
    if (hole !== null) return hole;
    return this.lgSizeUsed < loc.lg ? this.lgSizeUsed : null;
  }

  allocateFromHole(loc, lg) {
    let result;
    if (!this.isUsed) {
      result = 0;
      this.isUsed = true;
      this.lgSizeUsed = lg;
    } else if (lg >= this.lgSizeUsed) {
      this.holes.addHolesAtEnd(this.lgSizeUsed, 1, lg);
      this.lgSizeUsed = lg + 1;
      result = 1;
    } else {
      const hole = this.holes.tryAllocate(lg);
      if (hole !== null) {
        result = hole;
      } else {
        result = 2 ** (this.lgSizeUsed - lg);
        this.holes.addHolesAtEnd(lg, result + 1, this.lgSizeUsed);
        this.lgSizeUsed += 1;
      }
    }
    return loc.offset * 2 ** (loc.lg - lg) + result;
  }

  tryAllocateByExpanding(union, loc, lg) {
    if (!this.isUsed) {
      if (!union.tryExpandTo(loc, lg)) return null;
      this.isUsed = true;
      this.lgSizeUsed = lg;
      return loc.offset * 2 ** (loc.lg - lg);
    }
    const newSize = Math.max(this.lgSizeUsed, lg) + 1;
    if (!this.tryExpandUsage(union, loc, newSize, true)) return null;
    return loc.offset * 2 ** (loc.lg - lg) + this.holes.tryAllocate(lg);
  }

  tryExpand(union, loc, oldLg, oldOffset, factor) {
    if (oldOffset === 0 && this.lgSizeUsed === oldLg) return this.tryExpandUsage(union, loc, oldLg + factor, false);
    return this.holes.tryExpand(oldLg, oldOffset, factor);
  }

  tryExpandUsage(union, loc, desired, newHoles) {
    if (desired > loc.lg && !union.tryExpandTo(loc, desired)) return false;
    if (newHoles) this.holes.addHolesAtEnd(this.lgSizeUsed, 1, desired);
    this.lgSizeUsed = desired;
    return true;
  }
}

class Group {
  constructor(parent) {
    this.parent = parent;
    this.usage = [];
    this.pointerUsage = 0;
    this.hasMembers = false;
  }

  addMember() {
    if (!this.hasMembers) {
      this.hasMembers = true;
      this.parent.newGroupAddingFirstMember();
    }
  }

  addVoid() {
    this.addMember();
    this.parent.parent.addVoid();
  }

  addData(lg) {
    this.addMember();
    const locs = this.parent.dataLocations;
    let bestSize = Infinity, best = null;
    for (let i = 0; i < locs.length; i++) {
      if (this.usage.length === i) this.usage.push(new Usage());
      const hole = this.usage[i].smallestHoleAtLeast(locs[i], lg);
      if (hole !== null && hole < bestSize) {
        bestSize = hole;
        best = i;
      }
    }
    if (best !== null) return this.usage[best].allocateFromHole(locs[best], lg);
    for (let i = 0; i < locs.length; i++) {
      const r = this.usage[i].tryAllocateByExpanding(this.parent, locs[i], lg);
      if (r !== null) return r;
    }
    const r = this.parent.addNewDataLocation(lg);
    this.usage.push(new Usage(lg));
    return r;
  }

  addPointer() {
    this.addMember();
    const locs = this.parent.pointerLocations;
    if (this.pointerUsage < locs.length) return locs[this.pointerUsage++];
    this.pointerUsage++;
    return this.parent.addNewPointerLocation();
  }

  tryExpandData(oldLg, oldOffset, factor) {
    if (oldLg + factor > 6 || (oldOffset & ((1 << factor) - 1)) !== 0) return false;
    const locs = this.parent.dataLocations;
    for (let i = 0; i < this.usage.length; i++) {
      const loc = locs[i];
      if (loc.lg >= oldLg && Math.floor(oldOffset / 2 ** (loc.lg - oldLg)) === loc.offset) {
        const local = oldOffset - loc.offset * 2 ** (loc.lg - oldLg);
        return this.usage[i].tryExpand(this.parent, loc, oldLg, local, factor);
      }
    }
    throw new Error("capnp: tried to expand a field that was never allocated");
  }
}

// ---- entry points

// json.dumps(separators=(",", ":")) with exact 64-bit integers
function json(v) {
  if (v === null || v === undefined) return "null";
  if (typeof v === "bigint") return v.toString();
  if (typeof v !== "object") return JSON.stringify(v);
  if (Array.isArray(v)) return `[${v.map(json).join(",")}]`;
  return `{${Object.entries(v).map(([k, x]) => `${JSON.stringify(k)}:${json(x)}`).join(",")}}`;
}

// the key of the file an import names: relative to the importing file, or "/x" for x on the import path (where
// opendbc's car.capnp lives); null for files that only carry annotations
export function resolveImport(from, name) {
  if (/(^|\/)(c\+\+|java)\.capnp$/.test(name)) return null;
  const rooted = name.startsWith("/") || from.startsWith("/");
  const parts = name.startsWith("/") ? [] : from.split("/").filter(Boolean).slice(0, -1);
  for (const p of name.split("/")) {
    if (p === "..") parts.pop();
    else if (p && p !== ".") parts.push(p);
  }
  return (rooted ? "/" : "") + parts.join("/");
}

// the imports a file names, for fetching them before compiling
export function imports(text) {
  return [...text.matchAll(/\bimport\s+"([^"]+)"/g)].map((m) => m[1]);
}

// compiles files ({path: text}) from root's struct (Event by default) into schema.json's text
export function compile(files, root = "log.capnp", struct = "Event") {
  const c = new Compiler(files, resolveImport);
  const node = c.file(root).members.get(struct);
  if (node?.kind !== "struct") throw new Error(`capnp: no struct ${struct} in ${root}`);
  return c.emit(node);
}
