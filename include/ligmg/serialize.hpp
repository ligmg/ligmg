// Simple serialization library for mpi
#pragma once

#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <tuple>
#include <vector>

namespace serialize {

using buffer = std::string;

// our "interface" for encoding
// template<class T>
// void encode(std::streambuf& os, const T& t);

// our "interface" for decoding
// template<class T>
// void decode(std::streambuf& os, T& t);

// Helper functions:
// template<class T>
// std::stringbuf encoded(const T&)
//
// template<class T>
// T decoded<T>(const std::stringbuf&)

// helpers to encode pointers to arrays of data
// also works for single elements
template <class T>
inline void encode_ptr(std::streambuf &os, T *t, size_t size = 1) {
  os.sputn((char *)t, sizeof(T) * size);
}

template <class T>
inline void decode_ptr(std::streambuf &os, T *t, size_t size = 1) {
  os.sgetn((char *)t, sizeof(T) * size);
}

// serialize integral and floating point types
template <class T>
inline typename std::enable_if<std::is_integral<T>::value, void>::type
encode(std::streambuf &os, const T &p) {
  encode_ptr(os, &p);
}
template <class T>
inline typename std::enable_if<std::is_integral<T>::value, void>::type
decode(std::streambuf &os, T &p) {
  decode_ptr(os, &p);
}
template <class T>
inline typename std::enable_if<std::is_floating_point<T>::value, void>::type
encode(std::streambuf &os, const T &p) {
  encode_ptr(os, &p);
}
template <class T>
inline typename std::enable_if<std::is_floating_point<T>::value, void>::type
decode(std::streambuf &os, T &p) {
  decode_ptr(os, &p);
}

// STL container serializers
// Some of these functions have template specializations to increase performance
// when
// the entire structure exists as a contiguous piece of memory
// we need some prototypes TODO: clean up order of functions
// TODO: some pod types might not be contigous... ex: pair<int*>, we need to
// handle this
// Need to declare prototypes for all functions so they can call eachother when
// instantiated
template <class T> inline void decode(std::streambuf &os, std::set<T> &v);
template <class T, class S>
inline typename std::enable_if<std::is_pod<T>::value && std::is_pod<S>::value,
                               void>::type
encode(std::streambuf &os, const std::pair<T, S> &p);
template <class T, class S>
inline
    typename std::enable_if<!(std::is_pod<T>::value && std::is_pod<S>::value),
                            void>::type
    encode(std::streambuf &os, const std::pair<T, S> &p);
template <class T, class S>
inline typename std::enable_if<std::is_pod<T>::value && std::is_pod<S>::value,
                               void>::type
decode(std::streambuf &os, std::pair<T, S> &p);
template <class T, class S>
inline
    typename std::enable_if<!(std::is_pod<T>::value && std::is_pod<S>::value),
                            void>::type
    decode(std::streambuf &os, std::pair<T, S> &p);
template <class T>
inline typename std::enable_if<std::is_pod<T>::value, void>::type
encode(std::streambuf &os, const std::vector<T> &v);
template <class T>
inline typename std::enable_if<!std::is_pod<T>::value, void>::type
encode(std::streambuf &os, const std::vector<T> &v);
template <class T>
inline typename std::enable_if<std::is_pod<T>::value, void>::type
decode(std::streambuf &os, std::vector<T> &v);
template <class T>
inline typename std::enable_if<!std::is_pod<T>::value, void>::type
decode(std::streambuf &os, std::vector<T> &v);
template <class T>
inline typename std::enable_if<std::is_pod<T>::value, void>::type
encode(std::streambuf &os, const std::set<T> &v);
template <class T>
inline typename std::enable_if<!std::is_pod<T>::value, void>::type
encode(std::streambuf &os, const std::set<T> &v);
template <class T>
inline typename std::enable_if<std::is_pod<T>::value, void>::type
decode(std::streambuf &os, std::set<T> &v);
template <class T>
inline typename std::enable_if<!std::is_pod<T>::value, void>::type
decode(std::streambuf &os, std::set<T> &v);
template <class T, class S>
inline void encode(std::streambuf &os, const std::map<T, S> &v);
template <class T, class S>
inline void decode(std::streambuf &os, std::map<T, S> &v);
template <class T, class S>
inline void encode(std::streambuf &os, const std::multimap<T, S> &v);
template <class T, class S>
inline void decode(std::streambuf &os, std::multimap<T, S> &v);

// std::pair
// we pray that std::pair is contiguous memory
// TODO: is it?
template <class T, class S>
inline typename std::enable_if<std::is_pod<T>::value && std::is_pod<S>::value,
                               void>::type
encode(std::streambuf &os, const std::pair<T, S> &p) {
  encode_ptr(os, &p);
}

// specialization for non-contiguous mem
template <class T, class S>
inline
    typename std::enable_if<!(std::is_pod<T>::value && std::is_pod<S>::value),
                            void>::type
    encode(std::streambuf &os, const std::pair<T, S> &p) {
  encode(os, p.first);
  encode(os, p.second);
}

template <class T, class S>
inline typename std::enable_if<std::is_pod<T>::value && std::is_pod<S>::value,
                               void>::type
decode(std::streambuf &os, std::pair<T, S> &p) {
  decode_ptr(os, &p);
}

template <class T, class S>
inline
    typename std::enable_if<!(std::is_pod<T>::value && std::is_pod<S>::value),
                            void>::type
    decode(std::streambuf &os, std::pair<T, S> &p) {
  decode(os, p.first);
  decode(os, p.second);
}

// helper to encode a size plus an interator
template <class Iter>
inline void encode_iter(std::streambuf &os, size_t size, Iter begin, Iter end);

// std::vector

// specialized for contiguous memory
template <class T>
inline typename std::enable_if<std::is_pod<T>::value, void>::type
encode(std::streambuf &os, const std::vector<T> &v) {
  encode(os, v.size());
  encode_ptr(os, v.data(), v.size());
}

template <class T>
inline typename std::enable_if<!std::is_pod<T>::value, void>::type
encode(std::streambuf &os, const std::vector<T> &v) {
  encode_iter(os, v.size(), v.begin(), v.end());
}

template <class T>
inline typename std::enable_if<std::is_pod<T>::value, void>::type
decode(std::streambuf &os, std::vector<T> &v) {
  // read size of the vector
  size_t size;
  decode(os, size);
  v.resize(size); // we know the size, so we reserve space
  decode_ptr(os, v.data(), size);
}

template <class T>
inline typename std::enable_if<!std::is_pod<T>::value, void>::type
decode(std::streambuf &os, std::vector<T> &v) {
  v.clear();
  size_t size;
  decode(os, size);
  v.reserve(size);
  for (size_t i = 0; i < size; i++) {
    typename std::remove_reference<decltype(v)>::type::value_type t;
    decode(os, t);
    v.emplace_back(std::move(t));
  }
}

// std::set
template <class T>
inline void encode(std::streambuf &os, const std::set<T> &v) {
  encode_iter(os, v.size(), v.begin(), v.end());
}

template <class T> inline void decode_thing(std::streambuf &os, T &v) {
  v.clear();
  size_t size;
  decode(os, size);
  auto hint = v.end();
  for (size_t i = 0; i < size; i++) {
    typename std::remove_reference<decltype(v)>::type::value_type t;
    decode(os, t);
    hint = v.emplace_hint(hint, std::move(t));
  }
}

template <class T> inline void decode(std::streambuf &os, std::set<T> &v) {
  decode_thing(os, v);
}

// std::map
template <class T, class S>
inline void encode(std::streambuf &os, const std::map<T, S> &v) {
  encode_iter(os, v.size(), v.begin(), v.end());
}

template <class T, class S>
inline void decode(std::streambuf &os, std::map<T, S> &v) {
  decode_thing(os, v);
}

// std::multimap
template <class T, class S>
inline void encode(std::streambuf &os, const std::multimap<T, S> &v) {
  encode_iter(os, v.size(), v.begin(), v.end());
}

template <class T, class S>
inline void decode(std::streambuf &os, std::multimap<T, S> &v) {
  decode_thing(os, v);
}

template <class Iter>
inline void encode_iter(std::streambuf &os, size_t size, Iter begin, Iter end) {
  encode(os, size);
  for (; begin != end; ++begin) {
    encode(os, *begin);
  }
}

template <class T> inline buffer encoded(const T &x) {
  std::stringbuf s;
  encode(s, x);
  return s.str();
}

template <class T> inline T decoded(buffer &s) {
  T x;
  std::stringbuf ss(s);
  decode(ss, x);
  return x;
}
}
